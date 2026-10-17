`timescale 1ns/1ps
module tb_learning_pipe2_en_valid;
    reg clk=0, rst=1, en=0, in_valid=0;
    reg [7:0] d=0;
    wire [7:0] q1,q2;
    wire v1,v2;
    learning_pipe2_en_valid dut (
        .clk(clk),.rst(rst),.en(en),.d(d),.in_valid(in_valid),
        .q1(q1),.q2(q2),.v1(v1),.v2(v2)
    );
    always #5 clk=~clk;
    task step(input bit r,e, input [7:0] data_value, input bit valid_value,
              input [7:0] a, input bit av, input [7:0] b, input bit bv);
        @(negedge clk);
        rst=r; en=e; d=data_value; in_valid=valid_value;
        @(posedge clk); #1;
        if (q1 !== a || v1 !== av || q2 !== b || v2 !== bv)
            $fatal(1,"FAIL %0t: (%0d,%0b) (%0d,%0b)",$time,q1,v1,q2,v2);
        $display("CHECK %0t: en=%0b rst=%0b q1=%0d v1=%0b q2=%0d v2=%0b",
                 $time,en,rst,q1,v1,q2,v2);
    endtask
    initial begin
        $timeformat(-9,0," ns",0);
        $dumpfile("runs/learning_pipe2_en_valid.vcd");
        $dumpvars(0,tb_learning_pipe2_en_valid);
        @(posedge clk); #1;
        if ({q1,q2,v1,v2} !== 18'd0) $fatal(1,"Initial reset failed");
        step(0,1,5,1,   5,1,0,0);
        step(0,0,99,0,  5,1,0,0);
        step(0,1,8,1,   8,1,5,1);
        step(0,1,77,0, 77,0,8,1);
        step(0,0,200,1,77,0,8,1);
        step(0,0,0,0,  77,0,8,1);
        step(0,1,0,1,   0,1,77,0);
        step(0,1,255,1,255,1,0,1);
        step(1,0,99,1,  0,0,0,0);
        step(0,1,12,1, 12,1,0,0);
        step(0,1,42,0, 42,0,12,1);
        $display("PASS learning_pipe2_en_valid");
        $finish;
    end
endmodule
