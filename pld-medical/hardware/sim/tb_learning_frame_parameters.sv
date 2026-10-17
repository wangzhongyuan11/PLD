`timescale 1ns/1ps
module tb_learning_frame_parameters;
    parameter integer N=12, T=100;
    reg clk=0, rst=1, start=0, clear_done=0, in_valid=0, out_ready=0;
    reg [7:0] in_data=0;
    wire in_ready, out_valid, out_mask, frame_done, done_status;
    wire [16:0] input_count, pixel_count, target_count;
    wire [1:0] state;
    integer tx=0, rx=0, hits=0, cycles=0;
    reg completing;
    learning_frame_threshold #(.FRAME_PIXELS(N), .THRESHOLD(T)) dut (.*);
    always #5 clk=~clk;
    function [7:0] sample(input integer i);
        case (i%6)
            0: sample=0;
            1: sample=79;
            2: sample=80;
            3: sample=99;
            4: sample=100;
            5: sample=255;
        endcase
    endfunction
    initial begin
        @(posedge clk); #1;
        if (state!==0 || input_count!==0 || pixel_count!==0) $fatal(1,"Reset");
        @(negedge clk); rst=0; start=1;
        @(posedge clk); #1;
        if (state!==1 || input_count!==0) $fatal(1,"Start");
        // No waveform for large frames. The original directed TB provides one.
        while (rx<N) begin
            @(negedge clk);
            start=0; in_valid=1;
            in_data=(tx<N) ? sample(tx) : 8'd255; // offer one unwanted extra
            out_ready=(cycles%4!=1);
            @(posedge clk);
            completing=0;
            if (out_valid && out_ready) begin
                if (rx>=tx) $fatal(1,"Bypass or duplicate");
                if (out_mask !== (sample(rx)<T)) $fatal(1,"Wrong label at %0d",rx);
                if (out_mask) hits=hits+1;
                rx=rx+1;
                completing=(rx==N);
            end
            if (in_valid && in_ready) begin
                if (tx>=N) $fatal(1,"Extra input accepted");
                tx=tx+1;
            end
            #1;
            if (input_count!==tx || pixel_count!==rx || target_count!==hits)
                $fatal(1,"Counters");
            if (frame_done!==completing || done_status!==(rx==N)) $fatal(1,"Completion timing");
            if (dut.full !== (tx-rx==1)) $fatal(1,"Occupancy");
            if (dut.full && out_mask !== (sample(rx)<T)) $fatal(1,"Stored result");
            cycles=cycles+1;
            if (cycles>8*N+20) $fatal(1,"Timeout");
        end
        repeat (3) begin
            @(negedge clk); in_valid=1; in_data=255; out_ready=1;
            @(posedge clk);
            if (in_ready || out_valid) $fatal(1,"Transfer in DONE");
            #1;
            if (frame_done!==0 || done_status!==1 || pixel_count!==N || target_count!==hits)
                $fatal(1,"Result retention");
        end
        @(negedge clk); clear_done=1;
        @(posedge clk); #1;
        if (state!==0 || done_status!==0 || pixel_count!==N || target_count!==hits)
            $fatal(1,"Acknowledgement");
        $display("PASS parameters N=%0d T=%0d inputs=%0d outputs=%0d targets=%0d",N,T,tx,rx,hits);
        $finish;
    end
endmodule
