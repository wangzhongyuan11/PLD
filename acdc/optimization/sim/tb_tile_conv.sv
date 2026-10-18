`timescale 1ns/1ps
module tb_tile_conv #(parameter integer P=4,parameter bit WITH_WRITER=0,parameter bit WITH_LOADER=0);
  localparam integer WORDS=(P==4?4608:3584),AW=$clog2(WORDS);
  logic aclk=0,aresetn=0,start=0,busy,done;
  logic [15:0] cfg_oh=0,cfg_ow=0;
  logic [7:0] cfg_ci=0;
  logic [3:0] cfg_co=0;
  logic cfg_kernel3=0,cfg_relu=0,cfg_input_slot=0,cfg_weight_slot=0;
  logic [2:0] error;
  logic load_slot=0,load_valid=0,load_ready;
  logic [$clog2(P)-1:0] load_bank=0;
  logic [AW-1:0] load_addr=0;
  logic [63:0] load_data=0,weight_data=0;
  logic [7:0] load_keep=8'hff,weight_addr=0;
  logic weight_slot=0,weight_valid=0,weight_ready,param_slot=0,param_valid=0,param_ready;
  logic [2:0] weight_bank=0;
  logic [255:0] param_bias=0,param_multiplier=0;
  logic [31:0] m_data,m_tag;
  logic [3:0] m_keep;
  logic m_first,m_last,m_valid,m_ready,random_ready=0,writer_ready=0;
  logic writer_start=0,writer_done,writer_busy;
  logic [7:0] writer_error;
  bit writer_finished=0;
  integer checked_words,expected_words;
  logic loader_start=0,loader_done,loader_busy;
  logic [7:0] loader_error;
  logic l_weight_slot,l_weight_valid,l_param_slot,l_param_valid;
  logic [2:0] l_weight_bank;
  logic [7:0] l_weight_addr;
  logic [63:0] l_weight_data;
  logic [255:0] l_bias,l_multiplier;
  logic [70:0] expected[32768],item;
  int fd,ef,rc,ncommands,nexpected,op,received=0,tiles=0,cycles=0;
  int requests=0,run_started=0,expected_requests;
  int unsigned rng=32'hacdc3516;
  tile_conv #(.P(P)) dut(
    .weight_slot(WITH_LOADER?l_weight_slot:weight_slot),.weight_bank(WITH_LOADER?l_weight_bank:weight_bank),
    .weight_addr(WITH_LOADER?l_weight_addr:weight_addr),.weight_data(WITH_LOADER?l_weight_data:weight_data),
    .weight_valid(WITH_LOADER?l_weight_valid:weight_valid),.param_slot(WITH_LOADER?l_param_slot:param_slot),
    .param_valid(WITH_LOADER?l_param_valid:param_valid),.param_bias(WITH_LOADER?l_bias:param_bias),
    .param_multiplier(WITH_LOADER?l_multiplier:param_multiplier),.*);
  always #5 aclk=~aclk;
  function automatic int unsigned next_rng(input int unsigned v);
    v^=v<<13;v^=v>>17;v^=v<<5;return v;
  endfunction
  assign m_ready=WITH_WRITER?writer_ready:random_ready;
  always @(negedge aclk)begin rng=next_rng(rng);random_ready=aresetn&&(rng[1]||rng[3]);end
  if(WITH_WRITER)begin:writer_system
    logic [71:0] cmd;
    logic cmd_valid,cmd_ready,st_valid,st_ready,data_last,data_valid,data_ready;
    logic [7:0] st,data_keep;
    logic [63:0] data;
    tile_writer #(.P(P)) writer(.aclk,.aresetn,.start(writer_start),.cfg_width(cfg_ow),.cfg_height(cfg_oh),
      .cfg_full_address(32'h10000000+tiles*65536),.cfg_full_stride((cfg_ow+4)*8),
      .cfg_pool_address(32'h18000000+tiles*65536),.cfg_pool_stride((cfg_ow/2+3)*8),
      .cfg_pool(!cfg_ow[0]&&!cfg_oh[0]),.s_data(m_data),.s_tag(m_tag),.s_keep(m_keep),
      .s_first(m_first),.s_last(m_last),.s_valid(m_valid),.s_ready(writer_ready),
      .cmd_data(cmd),.cmd_valid,.cmd_ready,.status_data(st),.status_valid(st_valid),.status_ready(st_ready),
      .dma_data(data),.dma_keep(data_keep),.dma_last(data_last),.dma_valid(data_valid),.dma_ready(data_ready),
      .busy(writer_busy),.done(writer_done),.error(writer_error));
    datamover_sink sink(.aclk,.aresetn,.cmd_data(cmd),.cmd_valid,.cmd_ready,
      .data,.keep(data_keep),.last(data_last),.valid(data_valid),.ready(data_ready),
      .status_data(st),.status_valid(st_valid),.status_ready(st_ready),.checked_words,.expected_words);
  end
  if(WITH_LOADER)begin:loader_system
    logic [71:0] cmd;
    logic cmd_valid,cmd_ready,st_valid,st_ready,data_last,data_valid,data_ready;
    logic [7:0] st,data_keep;
    logic [63:0] data;
    integer supplied_words;
    parameter_loader loader(.aclk,.aresetn,.start(loader_start),
      .cfg_address(32'h08000000+tiles*65536),.cfg_ci,.cfg_kernel3,.cfg_slot(cfg_weight_slot),
      .busy(loader_busy),.done(loader_done),.error(loader_error),.cmd_data(cmd),.cmd_valid,.cmd_ready,
      .status_data(st),.status_valid(st_valid),.status_ready(st_ready),
      .dma_data(data),.dma_keep(data_keep),.dma_last(data_last),.dma_valid(data_valid),.dma_ready(data_ready),
      .weight_slot(l_weight_slot),.weight_bank(l_weight_bank),.weight_addr(l_weight_addr),
      .weight_data(l_weight_data),.weight_valid(l_weight_valid),.weight_ready,
      .param_slot(l_param_slot),.param_valid(l_param_valid),.param_bias(l_bias),.param_multiplier(l_multiplier),.param_ready);
    datamover_source source(.aclk,.aresetn,.cmd_data(cmd),.cmd_valid,.cmd_ready,
      .data,.keep(data_keep),.last(data_last),.valid(data_valid),.ready(data_ready),
      .status_data(st),.status_valid(st_valid),.status_ready(st_ready),.supplied_words);
  end
  always @(posedge aclk)if(aresetn)begin
    cycles++;
    if(WITH_WRITER&&writer_done)writer_finished=1;
    if(dut.request_valid&&dut.request_ready)requests++;
    if(m_valid&&m_ready)begin
      if(received>=nexpected)$fatal(1,"extra tile output");
      item=expected[received];
      if({m_tag,m_last,m_first,1'b0,m_keep}!==item[70:32])$fatal(1,"tile metadata mismatch P=%0d beat=%0d tag=%h keep=%h exp=%h",P,received,m_tag,m_keep,item[70:32]);
      for(int l=0;l<4;l++)if(m_keep[l]&&m_data[l*8+:8]!==item[l*8+:8])
        $fatal(1,"tile value mismatch P=%0d beat=%0d lane=%0d actual=%h expected=%h tag=%h",P,received,l,m_data,item[31:0],m_tag);
      received++;
    end
  end
  initial begin
    fd=$fopen("commands.txt","r");ef=$fopen("tile_expected.txt","r");
    if(!fd||!ef)$fatal(1,"tile vectors missing");
    rc=$fscanf(fd,"%d\n",ncommands);rc=$fscanf(ef,"%d\n",nexpected);
    if(nexpected>32768)$fatal(1,"too many outputs");
    for(int i=0;i<nexpected;i++)begin rc=$fscanf(ef,"%h\n",expected[i]);if(rc!=1)$fatal(1,"short expected");end
    repeat(4)@(negedge aclk);aresetn=1;
    for(int j=0;j<ncommands;j++)begin
      @(negedge aclk);load_valid=0;weight_valid=0;param_valid=0;start=0;
      rc=$fscanf(fd,"%h",op);if(rc!=1)$fatal(1,"short tile command");
      case(op)
        0:begin
          rc=$fscanf(fd,"%h %h %h %h\n",load_slot,load_bank,load_addr,load_data);
          if(rc!=4)$fatal(1,"bad input command");load_valid=1;
          do @(posedge aclk);while(!load_ready);
        end
        1:begin
          rc=$fscanf(fd,"%h %h %h %h\n",weight_slot,weight_bank,weight_addr,weight_data);
          if(rc!=4)$fatal(1,"bad weight command");weight_valid=1;
          do @(posedge aclk);while(!weight_ready);
        end
        2:begin
          rc=$fscanf(fd,"%h %h %h\n",param_slot,param_bias,param_multiplier);
          if(rc!=3)$fatal(1,"bad parameter command");param_valid=1;
          do @(posedge aclk);while(!param_ready);
        end
        3:begin
          rc=$fscanf(fd,"%h %h %h %h %h %h %h %h\n",cfg_ci,cfg_co,cfg_oh,cfg_ow,cfg_kernel3,cfg_relu,cfg_input_slot,cfg_weight_slot);
          if(rc!=8)$fatal(1,"bad run command");
          if(WITH_LOADER)begin
            loader_start=1;@(posedge aclk);@(negedge aclk);loader_start=0;
            wait(loader_done);@(negedge aclk);
            if(loader_error)$fatal(1,"parameter loader error=%h",loader_error);
          end
          start=1;
          requests=0;run_started=cycles;writer_finished=0;writer_start=WITH_WRITER;
          @(posedge aclk);@(negedge aclk);start=0;writer_start=0;
          wait(done);@(negedge aclk);
          if(error)$fatal(1,"tile error=%0d",error);
          if(WITH_WRITER)begin
            wait(writer_finished);@(negedge aclk);
            if(writer_error)$fatal(1,"writer error=%h",writer_error);
          end
          expected_requests=cfg_oh*((cfg_ow+P-1)/P)*((cfg_ci+7)/8)*(cfg_kernel3?9:1);
          if(requests!=expected_requests)$fatal(1,"missing/duplicate window taps %0d/%0d",requests,expected_requests);
          $display("TILE_RUN tile=%0d P=%0d ci=%0d co=%0d oh=%0d ow=%0d k=%0d cycles=%0d issues=%0d",tiles,P,cfg_ci,cfg_co,cfg_oh,cfg_ow,cfg_kernel3?3:1,cycles-run_started,requests);
          tiles++;
        end
        default:$fatal(1,"unknown command");
      endcase
    end
    @(negedge aclk);load_valid=0;weight_valid=0;param_valid=0;
    if(received!=nexpected)$fatal(1,"missing tile outputs %0d/%0d",received,nexpected);
    if(WITH_WRITER&&checked_words!=expected_words)$fatal(1,"missing addressed DDR writes %0d/%0d",checked_words,expected_words);
    cfg_ci=0;start=1;
    @(posedge aclk);@(negedge aclk);start=0;
    if(!done||busy||error!=1)$fatal(1,"invalid Ci was not rejected");
    @(negedge aclk);cfg_ci=192;cfg_co=8;cfg_oh=256;cfg_ow=256;cfg_kernel3=1;start=1;
    @(posedge aclk);@(negedge aclk);start=0;
    wait(done);@(negedge aclk);
    if(busy||error!=1)$fatal(1,"over-capacity tile was not rejected");
    aresetn=0;repeat(3)@(negedge aclk);aresetn=1;
    @(negedge aclk);if(error||busy||m_valid)$fatal(1,"tile status did not reset");
    if(WITH_WRITER)$display("TB_PASS tile_writer P=%0d addressed_words=%0d conv_pool_gap_stride_padding=PASS",P,checked_words);
    if(WITH_LOADER)$display("TB_PASS tile_parameter_system P=%0d DMA_parameters_conv_pool_write=PASS",P);
    $display("TB_PASS tile_conv P=%0d dense_tiles=%0d output_beats=%0d expected_errors=2 cycles=%0d",P,tiles,received,cycles);$finish;
  end
  initial begin #40000000;$fatal(1,"tile watchdog");end
endmodule
module tb_tile_conv_p4;
  tb_tile_conv #(.P(4)) test();
endmodule
module tb_tile_conv_p6;
  tb_tile_conv #(.P(6)) test();
endmodule
module tb_tile_writer_p4;
  tb_tile_conv #(.P(4),.WITH_WRITER(1)) test();
endmodule
module tb_tile_writer_p6;
  tb_tile_conv #(.P(6),.WITH_WRITER(1)) test();
endmodule
module tb_tile_parameter_p4;
  tb_tile_conv #(.P(4),.WITH_WRITER(1),.WITH_LOADER(1)) test();
endmodule
module tb_tile_parameter_p6;
  tb_tile_conv #(.P(6),.WITH_WRITER(1),.WITH_LOADER(1)) test();
endmodule
