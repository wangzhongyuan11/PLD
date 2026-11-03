`timescale 1ns/1ps
module tb_learning_threshold_count;
    reg clk=0, rst=1, in_valid=0, out_ready=0;
    reg [7:0] in_data=0;
    wire in_ready, out_mask, out_valid, frame_done, done_status;
    reg clear_done=0, expected_status=0;
    wire [16:0] pixel_count, target_count;
    reg expected [0:63];
    integer sent=0, received=0, targets=0;
    reg blocked=0, held_mask=0;
    integer done_pulses=0;
    learning_threshold_count dut (
        .clk(clk), .rst(rst), .in_data(in_data), .in_valid(in_valid),
        .in_ready(in_ready), .out_mask(out_mask), .out_valid(out_valid),
        .out_ready(out_ready), .pixel_count(pixel_count), .target_count(target_count),
        .frame_done(frame_done), .done_status(done_status), .clear_done(clear_done)
    );
    always #5 clk=~clk;

    task step(input bit r, iv, input [7:0] data_value, input bit ready,
              input bit clear_value=0);
        reg expected_done;
        begin
            @(negedge clk);
            rst=r; in_valid=iv; in_data=data_value; out_ready=ready;
            clear_done=clear_value;
            @(posedge clk);
            expected_done=0;
            if (rst) begin
                sent=0; received=0; targets=0; blocked=0;
            end else begin
                if (in_ready !== ((sent==received) || out_ready))
                    $fatal(1,"Input readiness mismatch");
                if (blocked && (out_valid !== 1'b1 || out_mask !== held_mask))
                    $fatal(1,"Output changed under backpressure");
                blocked=out_valid && !out_ready;
                held_mask=out_mask;
                // Consume old output BEFORE accepting new input: no bypass.
                if (out_valid && out_ready) begin
                    if (received >= sent) $fatal(1,"Unexpected output");
                    if (out_mask !== expected[received]) $fatal(1,"Mask/order mismatch");
                    received=received+1;
                    expected_done=(received==6);
                    if (out_mask) targets=targets+1;
                    $display("RECEIVE %0t mask=%b",$time,out_mask);
                end
                if (in_valid && in_ready) begin
                    expected[sent]=(in_data < 100);
                    sent=sent+1;
                end
            end
            if (rst) expected_status=0;
            else if (expected_done) expected_status=1;
            else if (clear_done) expected_status=0;
            #1;
            if (done_status !== expected_status) $fatal(1,"Sticky status mismatch at %0t",$time);
            if (frame_done !== expected_done) $fatal(1,"Done pulse mismatch at %0t",$time);
            if (frame_done) done_pulses=done_pulses+1;
            if (pixel_count !== received || target_count !== targets)
                $fatal(1,"Counter mismatch at %0t",$time);
            if (out_valid !== (sent!=received)) $fatal(1,"Occupancy mismatch");
            if (out_valid && out_mask !== expected[received]) $fatal(1,"Stored mask mismatch");
            $display("CHECK %0t valid=%b mask=%b pixels=%0d targets=%0d pulse=%b status=%b clear=%b",
                     $time,out_valid,out_mask,pixel_count,target_count,frame_done,done_status,clear_done);
        end
    endtask
    initial begin
        $timeformat(-9,0," ns",0);
        $dumpfile("runs/learning_threshold_count_status.vcd");
        $dumpvars(0,tb_learning_threshold_count);
        @(posedge clk); #1;
        if (out_valid !== 0 || pixel_count !== 0 || target_count !== 0 || frame_done !== 0 || done_status !== 0)
            $fatal(1,"Initial reset failed");
        // Previously verified Python image: 0,80,255,30,160,220.
        step(0,1,0,0);
        step(0,1,80,0);
        step(0,1,80,0);
        step(0,1,80,1);
        step(0,1,255,1);
        step(0,1,30,0);
        step(0,1,30,1);
        step(0,0,99,1);
        step(0,0,99,1);
        step(0,1,160,1);
        step(0,1,220,1);
        step(0,0,0,0); // Last result stalls: count=5, done must remain zero.
        step(0,0,0,0);
        step(0,0,0,1);
        if (sent!=6 || received!=6 || targets!=3) $fatal(1,"Tiny image totals");
        $display("PASS tiny_image masks=110100 pixels=6 targets=3");
        step(0,0,0,1); // Pulse clears; counters must retain the result.
        step(0,0,0,1);
        if (done_status !== 1) $fatal(1,"Late status read lost completion");
        step(0,0,0,1,1); // Acknowledge: status clears, results remain.
        step(0,0,0,1);
        step(1,0,0,0);
        // Threshold boundary and extremes, plus repeated zero labels.
        step(0,1,0,1);
        step(0,1,99,1);
        step(0,1,100,1);
        step(0,1,255,1);
        step(0,0,0,1);
        if (received!=4 || targets!=2) $fatal(1,"Boundary totals");
        $display("PASS boundary masks=1100 pixels=4 targets=2");
        step(0,1,80,0);
        step(1,1,80,0); // discard pending result; reset has priority
        step(0,0,0,1);
        step(0,1,100,1);
        step(0,0,0,1);
        if (received!=1 || targets!=0) $fatal(1,"Reset recovery totals");
        step(1,0,0,0);
        repeat (6) step(0,1,80,1);
        step(0,0,0,1,1); // Completion must win over simultaneous clear.
        if (received!=6 || targets!=6 || frame_done!==1) $fatal(1,"All-target frame");
        if (done_status !== 1) $fatal(1,"Completion did not win over clear");
        step(1,0,0,0,1); // Reset while done was high.
        if (done_pulses!=2) $fatal(1,"Expected exactly two complete frames");
        $display("PASS learning_threshold_count_status pulses=%0d",done_pulses);
        $finish;
    end
endmodule
