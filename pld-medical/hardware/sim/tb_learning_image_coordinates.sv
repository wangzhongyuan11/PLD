`timescale 1ns/1ps
module tb_learning_image_coordinates;
    reg clk=0, rst=1, start=0, clear_done=0, in_valid=0, out_ready=0;
    reg [7:0] cfg_threshold=100, in_data=0;
    wire [7:0] frame_threshold;
    wire in_ready, out_valid, out_mask, frame_done, done_status;
    wire [16:0] input_count, pixel_count, target_count;
    wire [1:0] state, in_col, out_col;
    wire in_row, out_row;
    integer frame, tx, rx, hits, cycles, next_index;
    integer stalls=0, bubbles=0, replaces=0;
    reg pending, push_now, pop_now, held;
    reg saved_mask, saved_row;
    reg [1:0] saved_col;
    learning_image_coordinates dut (.*);
    always #5 clk=~clk;
    initial begin
        $dumpfile("runs/learning_image_coordinates.vcd");
        $dumpvars(0,tb_learning_image_coordinates);
        @(posedge clk); #1;
        if ({in_row,in_col,out_row,out_col} !== 0) $fatal(1,"Reset coordinates");
        for (frame=0; frame<2; frame=frame+1) begin
            tx=0; rx=0; hits=0; cycles=0; pending=0;
            @(negedge clk);
            rst=0; start=1; clear_done=0; in_valid=0; out_ready=0;
            cfg_threshold=(frame==0) ? 100 : 170;
            @(posedge clk); #1;
            if (state!==1 || input_count!==0 || in_row!==0 || in_col!==0)
                $fatal(1,"Accepted start must reset input coordinates");
            while (rx<6) begin
                @(negedge clk);
                start=0;
                if (!pending && tx<6 && cycles%4!=1) pending=1;
                in_valid=pending;
                in_data=tx*50;
                out_ready=(cycles%5>=2);
                @(posedge clk);
                push_now=in_valid && in_ready;
                pop_now=out_valid && out_ready;
                held=out_valid && !out_ready;
                saved_row=out_row; saved_col=out_col; saved_mask=out_mask;
                if (held) stalls=stalls+1;
                if (!in_valid) bubbles=bubbles+1;
                if (push_now && pop_now) replaces=replaces+1;
                if (pop_now) begin
                    if (rx>=tx || out_row !== (rx/3==1) || out_col !== (rx%3))
                        $fatal(1,"Output coordinate/order mismatch rx=%0d",rx);
                    if (out_mask !== (rx*50 < cfg_threshold)) $fatal(1,"Mask mismatch");
                    $display("OUTPUT frame=%0d index=%0d coordinate=(%0d,%0d) mask=%b",
                             frame+1,rx,out_row,out_col,out_mask);
                    if (out_mask) hits=hits+1;
                    rx=rx+1;
                end
                if (push_now) begin
                    if (in_row !== (tx/3==1) || in_col !== (tx%3))
                        $fatal(1,"Input coordinate mismatch tx=%0d",tx);
                    tx=tx+1;
                    pending=0;
                end
                #1;
                next_index=(tx<6) ? tx : 5;
                if (in_row !== (next_index/3==1) || in_col !== (next_index%3))
                    $fatal(1,"Input cursor update/hold mismatch");
                if (held && (!out_valid || out_row!==saved_row ||
                             out_col!==saved_col || out_mask!==saved_mask))
                    $fatal(1,"Stalled output changed");
                if (input_count!==tx || pixel_count!==rx || target_count!==hits)
                    $fatal(1,"Counter mismatch");
                if (done_status !== (rx==6) || frame_done !== (pop_now && rx==6))
                    $fatal(1,"Completion mismatch");
                cycles=cycles+1;
                if (cycles>100) $fatal(1,"Timeout");
            end
            // DONE rejects additional input and start; clear only returns to WAIT.
            @(negedge clk); in_valid=1; in_data=42; start=1; clear_done=1;
            @(posedge clk);
            if (in_ready || out_valid) $fatal(1,"Unexpected transfer in DONE");
            #1;
            if (state!==0 || in_row!==1 || in_col!==2 || frame_done!==0)
                $fatal(1,"Clear must retain last coordinates and return to WAIT");
        end
        @(negedge clk); rst=1; start=0; clear_done=0;
        @(posedge clk); #1;
        if ({in_row,in_col,out_row,out_col} !== 0 || state!==0)
            $fatal(1,"Final reset");
        if (stalls==0 || bubbles==0 || replaces==0) $fatal(1,"Missing coverage");
        $display("PASS coordinates frames=2 stalls=%0d bubbles=%0d replacements=%0d",
                 stalls,bubbles,replaces);
        $finish;
    end
endmodule
