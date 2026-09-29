`timescale 1ns/1ps
module tb_tensor_tile_engine;
  logic aclk=0, aresetn=0, start=0;
  logic [1:0] cfg_op=0;
  logic [15:0] cfg_h=0,cfg_w=0;
  logic [6:0] cfg_channels=0,cfg_split=0;
  logic busy,done;
  logic [2:0] error;
  logic lut_we=0;
  logic [8:0] lut_addr=0;
  logic [7:0] lut_data=0;
  logic input_re,input_valid,output_we;
  logic [14:0] input_addr;
  logic [31:0] input_data;
  logic [2:0] output_bank;
  logic [12:0] output_addr;
  logic [7:0] output_data;
  logic load_we=0;
  logic [1:0] load_bank=0;
  logic [14:0] load_addr=0;
  logic [7:0] load_data=0;
  logic check_re=0,check_valid;
  logic [2:0] check_bank=0;
  logic [12:0] check_addr=0;
  logic [7:0] check_data;
  int fd,rc,op,a,b,c,d,e,cycles,runs=0,rejected=0,checks=0;
  logic verified=0;
  tensor_tile_engine dut(.*);
  // Exercise the real shared BRAM access, including the one-clock read latency.
  conv_tile_engine memory (
    .aclk,.aresetn,.start(1'b0),.cfg_in_h(16'd0),.cfg_in_w(16'd0),
    .cfg_ci(7'd0),.cfg_co(7'd0),.cfg_kernel3(1'b0),.busy(),.done(),.error(),
    .cfg_fused_concat(1'b0),.cfg_split(7'd0),
    .input_we(load_we),.input_bank(load_bank),.input_addr(load_addr),.input_data(load_data),
    .weight_we(1'b0),.weight_bank(5'd0),.weight_addr(11'd0),.weight_data(8'd0),
    .param_we(1'b0),.param_multiplier(1'b0),.param_channel(6'd0),.param_data(32'd0),
    .output_re(check_re),.output_bank(check_bank),.output_addr(check_addr),
    .output_data(check_data),.output_valid(check_valid),
    .aux_input_re(input_re),.aux_input_addr(input_addr),.aux_input_data(input_data),.aux_input_valid(input_valid),
    .aux_output_we(output_we),.aux_output_bank(output_bank),.aux_output_addr(output_addr),.aux_output_data(output_data)
  );
  always #5 aclk=~aclk;
  task automatic launch;
    start=1; @(negedge aclk); start=0;
  endtask
  initial begin
    repeat(4) @(negedge aclk); aresetn=1;
    fd=$fopen("../vectors/tensor.txt","r");
    if (!fd) $fatal(1,"missing tensor vectors");
    while (!$feof(fd)) begin
      rc=$fscanf(fd,"%h %h %h %h %h %h\n",op,a,b,c,d,e);
      if (rc==6) begin
        load_we=0; lut_we=0; check_re=0;
        case(op)
          0: begin cfg_h=a; cfg_w=b; cfg_channels=c; cfg_op=d; cfg_split=e; end
          1: begin load_we=1; load_bank=a; load_addr=b; load_data=c; end
          2: begin lut_we=1; lut_addr=a; lut_data=b; end
          3: begin
            launch();
            if (a==1) begin
              repeat(19) @(negedge aclk);
              if (!busy) $fatal(1,"reset case already complete");
              aresetn=0; repeat(3) @(negedge aclk);
              aresetn=1; repeat(3) @(negedge aclk);
              if (busy || done || output_we || input_re) $fatal(1,"reset failed");
              launch();
            end
            cycles=0;
            while (!done && cycles<3000000) begin @(negedge aclk); cycles++; end
            if (!done || busy || error!==b[2:0])
              $fatal(1,"tensor run=%0d done=%b busy=%b error=%0d expected=%0d",runs,done,busy,error,b);
            if (b==0) begin runs++; $display("phase tensor run=%0d cycles=%0d OK",runs,cycles); end
            else rejected++;
          end
          4: begin
            check_re=1; check_bank=a; check_addr=b; @(negedge aclk);
            if (!check_valid || check_data!==c[7:0])
              $fatal(1,"tensor run=%0d bank=%0d address=%0d actual=%h expected=%h",runs,a,b,check_data,c[7:0]);
            check_re=0; checks++;
          end
          7: begin
            if (runs!=a || rejected!=b || checks!=c || runs==0 || checks==0) $fatal(1,"missing vector checks");
            verified=1;
          end
          default: $fatal(1,"unknown opcode");
        endcase
        @(negedge aclk);
      end else if (!$feof(fd)) $fatal(1,"malformed vector");
    end
    $fclose(fd);
    if (!verified) $fatal(1,"missing final counts");
    $display("TB_PASS tb_tensor_tile_engine runs=%0d rejected=%0d output_bytes=%0d",runs,rejected,checks);
    $finish;
  end
  initial begin #100000000; $fatal(1,"watchdog"); end
endmodule
