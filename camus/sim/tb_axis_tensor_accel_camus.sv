`timescale 1ns/1ps
module tb_axis_tensor_accel_camus;
  logic aclk=0,aresetn=0;
  logic [63:0] s_axis_tdata=0,m_axis_tdata;
  logic [7:0] s_axis_tkeep=0,m_axis_tkeep;
  logic s_axis_tvalid=0,s_axis_tlast=0,s_axis_tready;
  logic m_axis_tvalid,m_axis_tlast,m_axis_tready=0;
  logic [11:0] s_axi_awaddr=0,s_axi_araddr=0;
  logic [2:0] s_axi_awprot=0,s_axi_arprot=0;
  logic s_axi_awvalid=0,s_axi_awready,s_axi_wvalid=0,s_axi_wready;
  logic [31:0] s_axi_wdata=0,s_axi_rdata;
  logic [3:0] s_axi_wstrb=0;
  logic [1:0] s_axi_bresp,s_axi_rresp;
  logic s_axi_bvalid,s_axi_bready=0,s_axi_arvalid=0,s_axi_arready,s_axi_rvalid,s_axi_rready=0;
  logic [72:0] expected_q[$],expected,held;
  logic held_valid=0;
  int fd,rc,op,ready_mode=2,runs=0,loads=0,errors=0,packets=0,bytes=0;
  logic [63:0] a,b,c,d,e,f,g;
  logic [31:0] value,status;
  int unsigned random_state=32'habcdef27;
  logic verified=0;
  axis_tensor_accel_camus dut(.*);
  always #5 aclk=~aclk;
  function automatic int unsigned random_next(input int unsigned state);
    state^=state<<13; state^=state>>17; state^=state<<5; return state;
  endfunction
  always @(negedge aclk) begin
    random_state=random_next(random_state);
    m_axis_tready=ready_mode==2 ? random_state[0] || random_state[1] : ready_mode==1;
  end
  always @(posedge aclk) begin
    if (aresetn) begin
      if (dut.state==dut.OUT_STREAM &&
          (dut.output_queued>4 || dut.output_queued+dut.output_read_pending>4 ||
           dut.output_issued-dut.output_count!=dut.output_queued+dut.output_read_pending))
        $fatal(1,"output RAM queue lost a byte or exceeded capacity");
      if (held_valid && (!m_axis_tvalid || {m_axis_tlast,m_axis_tkeep,m_axis_tdata}!==held))
        $fatal(1,"result changed under backpressure");
      held_valid=m_axis_tvalid && !m_axis_tready;
      held={m_axis_tlast,m_axis_tkeep,m_axis_tdata};
      if (m_axis_tvalid && m_axis_tready) begin
        if (expected_q.size()==0) $fatal(1,"unexpected result");
        expected=expected_q.pop_front();
        if ({m_axis_tlast,m_axis_tkeep,m_axis_tdata}!==expected)
          $fatal(1,"result mismatch packet=%0d byte=%0d actual=%h expected=%h",packets,bytes,{m_axis_tlast,m_axis_tkeep,m_axis_tdata},expected);
        for(int i=0;i<8;i++) if (m_axis_tkeep[i]) bytes++;
        if (m_axis_tlast) packets++;
      end
    end else held_valid=0;
  end
  task automatic aw(input logic [11:0] address,input int delay_cycles);
    repeat(delay_cycles) @(negedge aclk);
    s_axi_awaddr=address; s_axi_awvalid=1;
    do @(posedge aclk); while (!s_axi_awready);
    @(negedge aclk); s_axi_awvalid=0;
  endtask
  task automatic wd(input logic [31:0] data,input logic [3:0] strobe,input int delay_cycles);
    repeat(delay_cycles) @(negedge aclk);
    s_axi_wdata=data; s_axi_wstrb=strobe; s_axi_wvalid=1;
    do @(posedge aclk); while (!s_axi_wready);
    @(negedge aclk); s_axi_wvalid=0;
  endtask
  task automatic write_reg(input logic [11:0] address,input logic [31:0] data,input logic [3:0] strobe,input logic [1:0] response);
    fork aw(address,random_state[3:2]); wd(data,strobe,random_state[5:4]); join
    while (!s_axi_bvalid) @(negedge aclk);
    if (s_axi_bresp!==response) $fatal(1,"write response addr=%h actual=%h expected=%h",address,s_axi_bresp,response);
    repeat(2) @(negedge aclk);
    s_axi_bready=1; @(negedge aclk); s_axi_bready=0;
  endtask
  task automatic read_reg(input logic [11:0] address,output logic [31:0] result);
    s_axi_araddr=address; s_axi_arvalid=1;
    do @(posedge aclk); while (!s_axi_arready);
    @(negedge aclk); s_axi_arvalid=0;
    while (!s_axi_rvalid) @(negedge aclk);
    if (s_axi_rresp!=0) $fatal(1,"read SLVERR address=%h",address);
    result=s_axi_rdata;
    s_axi_rready=1; @(negedge aclk); s_axi_rready=0;
  endtask
  task automatic send_beat(input logic [63:0] data,input logic [7:0] keep,input logic last);
    s_axis_tdata=data; s_axis_tkeep=keep; s_axis_tlast=last; s_axis_tvalid=1;
    do @(posedge aclk); while (!s_axis_tready);
    @(negedge aclk); s_axis_tvalid=0;
    if (random_state[3:2]==0) repeat(2) @(negedge aclk);
  endtask
  task automatic wait_job(input int err,input logic [31:0] job,frame,received,sent);
    int polls;
    polls=0;
    do begin read_reg(12'h00c,status); polls++; end while (!status[1] && polls<1000000);
    if (!status[1] || status[0] || status[2]!=(err!=0)) $fatal(1,"job status mismatch job=%0d status=%h err=%0d",job,status,err);
    read_reg(12'h010,value); if (value!=err) $fatal(1,"error code job=%0d actual=%0d expected=%0d",job,value,err);
    read_reg(12'h03c,value); if (value!=frame) $fatal(1,"completed frame mismatch");
    read_reg(12'h040,value); if (value!=job) $fatal(1,"completed job mismatch");
    read_reg(12'h050,value); if (value!=received) $fatal(1,"RX bytes job=%0d actual=%0d expected=%0d",job,value,received);
    read_reg(12'h054,value); if (value!=sent) $fatal(1,"TX bytes job=%0d actual=%0d expected=%0d",job,value,sent);
    if (err!=0) errors++;
    else if (sent!=0) begin
      if (expected_q.size()!=0) $fatal(1,"DONE preceded output");
      runs++;
      read_reg(12'h058,value);
      $display("CAMUS_RUN run=%0d bytes=%0d cycles=%0d",runs,sent,value);
    end else loads++;
  endtask
  initial begin
    repeat(4) @(negedge aclk); aresetn=1;
    fd=$fopen("../vectors/accel.txt","r");
    if (!fd) $fatal(1,"missing vectors");
    while (!$feof(fd)) begin
      rc=$fscanf(fd,"%h %h %h %h %h %h %h %h\n",op,a,b,c,d,e,f,g);
      if (rc==8) begin
        case(op)
          0: write_reg(a[11:0],b[31:0],c[3:0],d[1:0]);
          1: send_beat(a,b[7:0],c[0]);
          2: wait_job(a,b[31:0],c[31:0],d[31:0],e[31:0]);
          3: expected_q.push_back({c[0],b[7:0],a});
          5: begin read_reg(a[11:0],value); if ((value&c[31:0])!==(b[31:0]&c[31:0])) $fatal(1,"readback addr=%h actual=%h expected=%h mask=%h",a,value,b,c); end
          6: repeat(a) @(negedge aclk);
          7: ready_mode=a;
          8: wait(m_axis_tvalid);
          10: begin
            // One explicit busy-write test loads input outside Writer.load().
            if (runs!=a || loads!=b || expected_q.size()!=0 || packets!=runs || errors!=c)
              $fatal(1,"missing checks runs=%0d loads=%0d packets=%0d errors=%0d",runs,loads,packets,errors);
            verified=1;
          end
          default: $fatal(1,"unknown vector opcode");
        endcase
        @(negedge aclk);
      end else if (!$feof(fd)) $fatal(1,"malformed vector");
    end
    if (!verified) $fatal(1,"no final counts");
    $display("TB_PASS tb_axis_tensor_accel_camus runs=%0d loads=%0d errors=%0d bytes=%0d",runs,loads,errors,bytes); $finish;
  end
  initial begin #500000000; $fatal(1,"watchdog state=%0d job=%0d",dut.state,dut.job_id); end
endmodule
