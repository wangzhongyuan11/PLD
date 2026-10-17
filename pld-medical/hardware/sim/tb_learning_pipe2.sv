`timescale 1ns/1ps
module tb_learning_pipe2;
    reg clk = 0;
    reg rst = 1;
    reg en = 0;
    reg [7:0] d = 0;
    wire [7:0] q1, q2;
    learning_pipe2 dut (.clk(clk), .rst(rst), .en(en), .d(d), .q1(q1), .q2(q2));
    always #5 clk = ~clk;

    task step(input bit reset_value, input bit enable_value,
              input [7:0] data_value, input [7:0] expected1, input [7:0] expected2);
        @(negedge clk);
        rst = reset_value; en = enable_value; d = data_value;
        @(posedge clk); #1;
        if (q1 !== expected1 || q2 !== expected2)
            $fatal(1, "FAIL at %0t: q1=%0d q2=%0d expected=%0d,%0d",
                   $time, q1, q2, expected1, expected2);
        $display("CHECK %0t: q1=%0d q2=%0d", $time, q1, q2);
    endtask

    initial begin
        $timeformat(-9, 0, " ns", 0);
        $dumpfile("runs/learning_pipe2.vcd");
        $dumpvars(0, tb_learning_pipe2);
        @(posedge clk); #1;
        if (q1 !== 0 || q2 !== 0) $fatal(1, "Initial reset failed");
        step(0, 1, 10, 10, 0);
        step(0, 1, 20, 20, 10);
        step(0, 0, 99, 20, 10);
        step(0, 1, 30, 30, 20);
        step(1, 1, 255, 0, 0);
        step(0, 1, 255, 255, 0);
        step(0, 1, 0, 0, 255);
        $display("PASS learning_pipe2");
        $finish;
    end
endmodule
