`timescale 1ns/1ps
module tb_learning_reg8;
    reg clk = 0;
    reg rst = 1;
    reg en = 0;
    reg [7:0] d = 0;
    wire [7:0] q;

    learning_reg8 dut (.clk(clk), .rst(rst), .en(en), .d(d), .q(q));
    always #5 clk = ~clk;

    task check(input [7:0] expected);
        if (q !== expected)
            $fatal(1, "FAIL time=%0t expected=%0d actual=%0d", $time, expected, q);
        $display("CHECK time=%0t q=%0d", $time, q);
    endtask

    initial begin
        $dumpfile("runs/learning_reg8.vcd");
        $dumpvars(0, tb_learning_reg8);
        // Reset at the first rising edge establishes a known state.
        @(posedge clk); #1; check(0);
        @(negedge clk); rst = 0; en = 1; d = 12;
        @(posedge clk); #1; check(12);
        @(negedge clk); en = 0; d = 99;
        @(posedge clk); #1; check(12);
        @(negedge clk); rst = 1; en = 1; d = 35;
        @(posedge clk); #1; check(0);
        @(negedge clk); rst = 0; en = 1; d = 200;
        @(posedge clk); #1; check(200);
        // Input and synchronous reset changes must not update q between edges.
        #1; d = 8; rst = 1;
        #1; check(200);
        @(posedge clk); #1; check(0);
        @(negedge clk); rst = 0; en = 1; d = 255;
        @(posedge clk); #1; check(255);
        @(negedge clk); d = 0;
        @(posedge clk); #1; check(0);
        $display("PASS learning_reg8");
        $finish;
    end
endmodule
