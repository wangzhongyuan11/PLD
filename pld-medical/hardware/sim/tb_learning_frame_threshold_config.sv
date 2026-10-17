`timescale 1ns/1ps
module tb_learning_frame_threshold_config;
    reg clk=0, rst=1, start=0, clear_done=0, in_valid=0, out_ready=0;
    reg [7:0] in_data=90, cfg_threshold=0;
    wire [7:0] frame_threshold;
    wire in_ready, out_valid, out_mask, frame_done, done_status;
    wire [16:0] input_count, pixel_count, target_count;
    wire [1:0] state;
    integer frame_no, tx, rx, hits, cycles, pulses=0;
    reg [7:0] initial_threshold;
    reg expected_label, completing;
    learning_frame_threshold_config dut (.*);
    always #5 clk=~clk;
    initial begin
        $timeformat(-9,0," ns",0);
        $dumpfile("runs/learning_frame_threshold_config.vcd");
        $dumpvars(0,tb_learning_frame_threshold_config);
        @(posedge clk); #1;
        if (frame_threshold!==0 || state!==0) $fatal(1,"Reset");
        for (frame_no=0; frame_no<2; frame_no=frame_no+1) begin
            initial_threshold=(frame_no==0) ? 100 : 80;
            expected_label=(frame_no==0);
            tx=0; rx=0; hits=0; cycles=0;
            @(negedge clk);
            rst=0; cfg_threshold=initial_threshold;
            start=1; clear_done=0; in_valid=0; out_ready=1;
            @(posedge clk); #1;
            if (state!==1 || frame_threshold!==initial_threshold || input_count!==0)
                $fatal(1,"Start must capture configuration");
            while (rx<6) begin
                @(negedge clk);
                // Extra start in RUN must not recapture changing configuration.
                start=(cycles==2);
                cfg_threshold=(cycles>=2) ? ((frame_no==0) ? 80 : 100) : initial_threshold;
                in_valid=(tx<6); in_data=90;
                out_ready=(cycles%3!=1);
                @(posedge clk);
                completing=0;
                if (out_valid && out_ready) begin
                    if (rx>=tx || out_mask!==expected_label) $fatal(1,"Mixed frame thresholds");
                    rx=rx+1;
                    if (out_mask) hits=hits+1;
                    completing=(rx==6);
                    $display("RECEIVE %0t frame=%0d gray=90 label=%b cfg=%0d active=%0d",
                             $time,frame_no+1,out_mask,cfg_threshold,frame_threshold);
                end
                if (in_valid && in_ready) tx=tx+1;
                #1;
                if (frame_threshold!==initial_threshold) $fatal(1,"Active threshold changed in RUN");
                if (input_count!==tx || pixel_count!==rx || target_count!==hits)
                    $fatal(1,"Counts mismatch");
                if (frame_done!==completing || done_status!==(rx==6)) $fatal(1,"Completion");
                if (dut.full !== (tx-rx==1)) $fatal(1,"Occupancy");
                if (dut.full && out_mask!==expected_label) $fatal(1,"Stored label");
                if (frame_done) pulses=pulses+1;
                cycles=cycles+1;
                if (cycles>40) $fatal(1,"Timeout");
            end
            if (tx!=6 || hits!=(frame_no==0 ? 6 : 0)) $fatal(1,"Frame totals");
            $display("PASS frame=%0d latched=%0d pixels=%0d targets=%0d",
                     frame_no+1,frame_threshold,pixel_count,target_count);
            @(negedge clk); start=1; cfg_threshold=17; in_valid=0;
            @(posedge clk); #1;
            if (state!==2 || frame_threshold!==initial_threshold || frame_done!==0)
                $fatal(1,"DONE must ignore start/config changes");
            @(negedge clk); start=0; clear_done=1;
            @(posedge clk); #1;
            if (state!==0 || target_count!==hits || frame_threshold!==initial_threshold)
                $fatal(1,"Acknowledgement must retain result/configuration");
        end
        @(negedge clk); rst=1; start=1; cfg_threshold=255;
        @(posedge clk); #1;
        if (state!==0 || frame_threshold!==0 || pixel_count!==0) $fatal(1,"Reset priority");
        if (pulses!=2) $fatal(1,"Pulse count");
        $display("PASS learning_frame_threshold_config frames=2");
        $finish;
    end
endmodule
