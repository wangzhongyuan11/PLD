`timescale 1ns/1ps
module tb_learning_frame_threshold;
    localparam WAIT=0, RUN=1, DONE=2;
    reg clk=0, rst=1, start=0, clear_done=0, in_valid=0, out_ready=0;
    reg [7:0] in_data=0;
    wire in_ready, out_valid, out_mask, frame_done, done_status;
    wire [16:0] input_count, pixel_count, target_count;
    wire [1:0] state;
    reg expected [0:15];
    integer written=0, read_count=0, completed=0;
    learning_frame_threshold dut (.*);
    always #5 clk=~clk;

    task step(input bit r,s,c,iv, input [7:0] data_value, input bit ready,
              input integer want_state, want_in, want_out, want_target,
              input bit want_done);
        begin
            @(negedge clk);
            rst=r; start=s; clear_done=c; in_valid=iv; in_data=data_value; out_ready=ready;
            @(posedge clk);
            if (rst || (state==WAIT && start)) begin
                if (in_ready || out_valid) $fatal(1,"Transfer during reset/start");
                written=0; read_count=0;
            end else begin
                if (out_valid && out_ready) begin
                    if (read_count>=written) $fatal(1,"Unexpected output or bypass");
                    if (out_mask !== expected[read_count]) $fatal(1,"Wrong label/order");
                    $display("RECEIVE %0t mask=%b",$time,out_mask);
                    read_count=read_count+1;
                end
                if (in_valid && in_ready) begin
                    if (written>=6) $fatal(1,"Seventh input accepted");
                    expected[written]=(in_data<100);
                    written=written+1;
                end
            end
            #1;
            if (state !== want_state || input_count !== want_in ||
                pixel_count !== want_out || target_count !== want_target ||
                frame_done !== want_done || done_status !== (want_state==DONE))
                $fatal(1,"State/count/status mismatch at %0t",$time);
            if (written!=want_in || read_count!=want_out) $fatal(1,"Handshake totals mismatch");
            if (dut.full !== (want_in-want_out==1)) $fatal(1,"Slot occupancy mismatch");
            if (dut.full && out_mask !== expected[read_count]) $fatal(1,"Pending label changed");
            if (frame_done) completed=completed+1;
            $display("CHECK %0t state=%0d in=%0d out=%0d target=%0d pulse=%b status=%b",
                     $time,state,input_count,pixel_count,target_count,frame_done,done_status);
        end
    endtask
    initial begin
        $timeformat(-9,0," ns",0);
        $dumpfile("runs/learning_frame_threshold.vcd");
        $dumpvars(0,tb_learning_frame_threshold);
        @(posedge clk); #1;
        if (state!==WAIT || input_count!==0 || done_status!==0) $fatal(1,"Initial reset");
        step(0,0,0,1,0,  1, WAIT,0,0,0,0); // data offered before start
        step(0,1,0,1,0,  1, RUN, 0,0,0,0); // start accepts no pixel
        step(0,0,0,1,0,  0, RUN, 1,0,0,0);
        step(0,1,1,1,80, 0, RUN, 1,0,0,0); // ignore start/clear while RUN
        step(0,0,0,1,80, 1, RUN, 2,1,1,0);
        step(0,0,0,1,255,1, RUN, 3,2,2,0);
        step(0,0,0,1,30, 1, RUN, 4,3,2,0);
        step(0,0,0,1,160,1, RUN, 5,4,3,0);
        step(0,0,0,1,220,1, RUN, 6,5,3,0);
        step(0,0,0,1,99, 0, RUN, 6,5,3,0); // refuse seventh, last output stalls
        step(0,1,1,1,99, 1, DONE,6,6,3,1); // completion wins over clear
        step(0,1,0,1,99, 1, DONE,6,6,3,0); // start ignored in DONE
        step(0,1,1,1,99, 1, WAIT,6,6,3,0); // acknowledge only, not restart
        step(0,0,0,1,99, 1, WAIT,6,6,3,0);
        step(0,1,0,1,99, 1, RUN, 0,0,0,0); // second frame without reset
        step(0,0,0,1,99, 1, RUN, 1,0,0,0); // held sample finally accepted
        step(0,0,0,1,100,1, RUN, 2,1,1,0);
        step(0,0,0,1,0,  1, RUN, 3,2,1,0);
        step(0,0,0,0,77, 1, RUN, 3,3,2,0); // bubble drains slot
        step(0,0,0,0,77, 1, RUN, 3,3,2,0);
        step(0,0,0,1,255,1, RUN, 4,3,2,0);
        step(0,0,0,1,80, 1, RUN, 5,4,2,0);
        step(0,0,0,1,160,1, RUN, 6,5,3,0);
        step(0,0,0,0,0,  1, DONE,6,6,3,1);
        if (completed!=2) $fatal(1,"Expected two completed frames");
        step(1,1,1,0,0,  1, WAIT,0,0,0,0); // reset priority in DONE
        step(0,1,0,1,80, 1, RUN, 0,0,0,0);
        step(0,0,0,1,80, 0, RUN, 1,0,0,0);
        step(1,1,1,0,0,  0, WAIT,0,0,0,0); // reset discards pending label
        $display("PASS learning_frame_threshold completed=%0d",completed);
        $finish;
    end
endmodule
