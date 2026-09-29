`timescale 1ns/1ps
module tb_conv_tile_engine;
  logic aclk=0, aresetn=0, start=0;
  logic [15:0] cfg_in_h=0, cfg_in_w=0;
  logic [6:0] cfg_ci=0, cfg_co=0;
  logic cfg_kernel3=0, cfg_fused_concat=0, busy, done;
  logic [6:0] cfg_split=0;
  logic [2:0] error;
  logic input_we=0, weight_we=0, param_we=0, param_multiplier=0, output_re=0;
  logic [1:0] input_bank=0;
  logic [14:0] input_addr=0;
  logic [7:0] input_data=0, weight_data=0, output_data;
  logic [4:0] weight_bank=0;
  logic [10:0] weight_addr=0;
  logic [5:0] param_channel=0;
  logic [31:0] param_data=0;
  logic [2:0] output_bank=0;
  logic [12:0] output_addr=0;
  logic output_valid;
  logic aux_input_re=0, aux_output_we=0, aux_input_valid;
  logic [14:0] aux_input_addr=0;
  logic [31:0] aux_input_data;
  logic [2:0] aux_output_bank=0;
  logic [12:0] aux_output_addr=0;
  logic [7:0] aux_output_data=0;
  int fd, rc, op, a, b, c, d, e, checks=0, runs=0, rejected=0, cycles;
  logic verified=0;
  conv_tile_engine dut(.*);
  always #5 aclk=~aclk;

  task automatic clear_ports;
    input_we=0; weight_we=0; param_we=0; output_re=0; start=0;
  endtask

  task automatic launch;
    start=1;
    @(negedge aclk);
    start=0;
  endtask

  initial begin
    repeat(4) @(negedge aclk);
    aresetn=1;
    fd=$fopen("../vectors/tile.txt","r");
    if (!fd) $fatal(1,"missing tile vectors");
    while (!$feof(fd)) begin
      rc=$fscanf(fd,"%h %h %h %h %h %h\n",op,a,b,c,d,e);
      if (rc==6) begin
        clear_ports();
        case (op)
          0: begin cfg_in_h=a; cfg_in_w=b; cfg_ci=c; cfg_co=d; cfg_kernel3=e;
                   cfg_fused_concat=0; cfg_split=0; end
          // Fused decoder-input convolution: a=ih,b=iw,c=ci,d=co,e=split.
          8: begin cfg_in_h=a; cfg_in_w=b; cfg_ci=c; cfg_co=d;
                   cfg_kernel3=1; cfg_fused_concat=1; cfg_split=e; end
          // Same, but with kernel3=0 to prove fused rejection.
          9: begin cfg_in_h=a; cfg_in_w=b; cfg_ci=c; cfg_co=d;
                   cfg_kernel3=0; cfg_fused_concat=1; cfg_split=e; end
          1: begin input_we=1; input_bank=a; input_addr=b; input_data=c; end
          2: begin weight_we=1; weight_bank=a; weight_addr=b; weight_data=c; end
          3,4: begin param_we=1; param_multiplier=(op==4); param_channel=a; param_data=b; end
          5: begin
            launch();
            if (a==1) begin
              // Interrupt a partially accumulated dot. RAM persists; valid pipelines reset.
              repeat(20) @(negedge aclk);
              if (!busy) $fatal(1,"reset case completed too early");
              aresetn=0;
              repeat(3) @(negedge aclk);
              aresetn=1;
              repeat(3) @(negedge aclk);
              if (busy || done || dut.read_valid || dut.quant_valid || dut.mac_valid)
                $fatal(1,"reset did not flush control/valid pipeline");
              launch();
            end
            cycles=0;
            while (!done && cycles<3000000) begin @(negedge aclk); cycles++; end
            if (!done || busy || error!==b[2:0])
              $fatal(1,"run=%0d timeout/error cycles=%0d done=%b busy=%b error=%0d expected=%0d",runs,cycles,done,busy,error,b);
            if (b==0) begin runs++; $display("phase tile run=%0d cycles=%0d OK",runs,cycles); end
            else rejected++;
          end
          6: begin
            output_re=1; output_bank=a; output_addr=b;
            @(negedge aclk);
            if (!output_valid || output_data!==c[7:0])
              $fatal(1,"output run=%0d bank=%0d address=%0d actual=%h expected=%h",runs,a,b,output_data,c[7:0]);
            output_re=0; checks++;
          end
          7: begin
            if (runs!=a || rejected!=b || checks!=c || runs==0 || checks==0)
              $fatal(1,"missing vectors runs=%0d rejected=%0d checks=%0d",runs,rejected,checks);
            verified=1;
          end
          default: $fatal(1,"invalid vector opcode");
        endcase
        @(negedge aclk);
      end else if (!$feof(fd)) $fatal(1,"malformed tile vector");
    end
    $fclose(fd);
    if (!verified) $fatal(1,"missing final vector counts");
    $display("TB_PASS tb_conv_tile_engine runs=%0d rejected=%0d output_bytes=%0d",runs,rejected,checks);
    $finish;
  end
  initial begin #2000000000; $fatal(1,"watchdog"); end
endmodule
