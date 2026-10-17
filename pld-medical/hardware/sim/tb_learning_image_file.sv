`timescale 1ns/1ps
module tb_learning_image_file;
    parameter integer N=6;
    reg clk=0, rst=1, start=0, clear_done=0, in_valid=0, out_ready=0;
    reg [7:0] cfg_threshold=0, in_data=0;
    wire [7:0] frame_threshold;
    wire in_ready, out_valid, out_mask, frame_done, done_status;
    wire [16:0] input_count, pixel_count, target_count;
    wire [1:0] state;
    reg [7:0] pixels [0:N-1];
    string input_path, output_path, wave_path;
    integer threshold_value, output_file, tx=0, rx=0, targets=0, cycles=0;
    reg complete_now;
    learning_frame_threshold_config #(.FRAME_PIXELS(N)) dut (.*);
    always #5 clk=~clk;
    initial begin
        if (!$value$plusargs("INPUT=%s",input_path) ||
            !$value$plusargs("OUTPUT=%s",output_path) ||
            !$value$plusargs("WAVE=%s",wave_path) ||
            !$value$plusargs("THRESHOLD=%d",threshold_value))
            $fatal(1,"Missing file/threshold arguments");
        if (threshold_value<0 || threshold_value>255) $fatal(1,"Invalid threshold");
        $readmemh(input_path,pixels);
        for (integer i=0;i<N;i=i+1)
            if (^pixels[i]===1'bx) $fatal(1,"Missing or unknown input pixel %0d",i);
        output_file=$fopen(output_path,"w");
        if (!output_file) $fatal(1,"Cannot open output file");
        $dumpfile(wave_path);
        $dumpvars(0,tb_learning_image_file);
        @(posedge clk); #1;
        @(negedge clk); rst=0; start=1; cfg_threshold=threshold_value;
        @(posedge clk); #1;
        if (state!==1 || input_count!==0 || frame_threshold!==threshold_value)
            $fatal(1,"Start/configuration mismatch");
        while (rx<N) begin
            @(negedge clk);
            start=0;
            in_valid=(tx<N);
            if (tx<N) in_data=pixels[tx];
            out_ready=(cycles%4!=1); // periodic backpressure
            @(posedge clk);
            complete_now=0;
            if (out_valid && out_ready) begin
                if (rx>=tx) $fatal(1,"Duplicate output or empty bypass");
                if (out_mask!==0 && out_mask!==1) $fatal(1,"Unknown label");
                $fdisplay(output_file,"%0d",out_mask);
                if (out_mask) targets=targets+1;
                rx=rx+1;
                complete_now=(rx==N);
            end
            // Advance input file position only after input handshake.
            if (in_valid && in_ready) tx=tx+1;
            #1;
            if (input_count!==tx || pixel_count!==rx || target_count!==targets)
                $fatal(1,"Counter mismatch");
            if (frame_done!==complete_now || done_status!==(rx==N))
                $fatal(1,"Completion mismatch");
            cycles=cycles+1;
            if (cycles>8*N+20) $fatal(1,"Timeout");
        end
        $fclose(output_file);
        if (tx!=N) $fatal(1,"Input count mismatch");
        @(negedge clk); in_valid=0; out_ready=1;
        @(posedge clk); #1;
        if (out_valid!==0 || frame_done!==0 || done_status!==1)
            $fatal(1,"Unexpected trailing output/status");
        $display("PASS file_stream N=%0d threshold=%0d targets=%0d",N,threshold_value,targets);
        $finish;
    end
endmodule
