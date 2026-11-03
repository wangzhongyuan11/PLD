`timescale 1ns/1ps
module tb_rgb_to_gray;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;
    reg in_valid = 0;
    reg [7:0] r = 0, g = 0, b = 0;
    wire out_valid;
    wire [7:0] gray;
    rgb_to_gray dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid),
                     .r(r), .g(g), .b(b), .out_valid(out_valid), .gray(gray));
    task check_pixel(input [7:0] rr, gg, bb, expected);
        begin
            @(negedge clk);
            in_valid = 1; r = rr; g = gg; b = bb;
            @(posedge clk); #1;
            if (out_valid !== 1'b1 || gray !== expected)
                $fatal(1, "gray mismatch: got %d expected %d", gray, expected);
        end
    endtask
    initial begin
        @(posedge clk); #1;
        if (out_valid !== 0 || gray !== 0) $fatal(1, "reset failed");
        @(negedge clk); rst_n = 1;
        check_pixel(0, 0, 0, 0);
        check_pixel(255, 255, 255, 255);
        check_pixel(255, 0, 0, 77);
        check_pixel(0, 255, 0, 149);
        check_pixel(0, 0, 255, 29);
        @(negedge clk); in_valid = 0;
        @(posedge clk); #1;
        if (out_valid !== 0) $fatal(1, "valid bubble failed");
        @(negedge clk); rst_n = 0;
        @(posedge clk); #1;
        if (out_valid !== 0 || gray !== 0) $fatal(1, "second reset failed");
        $display("PASS rgb_to_gray basic vectors");
        $finish;
    end
endmodule
