`timescale 1ns/1ps
module tb_learning_pipe2_valid;
    reg clk = 0, rst = 1, in_valid = 0;
    reg [7:0] d = 0;
    wire [7:0] q1, q2;
    wire v1, v2;
    reg [7:0] model_q1 = 0;
    reg model_v1 = 0;
    learning_pipe2_valid dut (
        .clk(clk), .rst(rst), .d(d), .in_valid(in_valid),
        .q1(q1), .q2(q2), .v1(v1), .v2(v2)
    );
    always #5 clk = ~clk;
    task step(input bit r, input [7:0] data_value, input bit valid_value);
        reg [7:0] expected_q2;
        reg expected_v2;
        @(negedge clk);
        rst = r; d = data_value; in_valid = valid_value;
        expected_q2 = r ? 0 : model_q1;
        expected_v2 = r ? 0 : model_v1;
        @(posedge clk); #1;
        if (q1 !== (r ? 8'd0 : data_value) || v1 !== (r ? 1'b0 : valid_value)
            || q2 !== expected_q2 || v2 !== expected_v2)
            $fatal(1, "Mismatch at %0t", $time);
        model_q1 = r ? 0 : data_value;
        model_v1 = r ? 0 : valid_value;
        $display("CHECK %0t d=%0d in_valid=%0b q2=%0d v2=%0b", $time,d,in_valid,q2,v2);
    endtask
    initial begin
        $timeformat(-9, 0, " ns", 0);
        $dumpfile("runs/learning_pipe2_valid.vcd");
        $dumpvars(0, tb_learning_pipe2_valid);
        @(posedge clk); #1;
        if ({q1,q2,v1,v2} !== 18'd0) $fatal(1,"Initial reset failed");
        step(0,5,1);
        step(0,99,0);
        step(0,0,1);
        step(0,42,0);
        step(0,255,1);
        step(0,17,1);
        step(0,33,0);
        step(0,44,0);
        step(0,88,1);
        step(1,77,1);
        step(0,12,0);
        step(0,34,0);
        $display("PASS learning_pipe2_valid");
        $finish;
    end
endmodule
