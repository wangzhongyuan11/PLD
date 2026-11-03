`timescale 1ns/1ps
module tb_learning_ready_valid_buffer;
    reg clk=0, rst=1, in_valid=0, out_ready=0;
    reg [7:0] in_data=0;
    wire in_ready, out_valid;
    wire [7:0] out_data;
    integer sent=0, received=0;
    reg [7:0] expected [0:63];
    learning_ready_valid_buffer dut (
        .clk(clk), .rst(rst), .in_data(in_data), .in_valid(in_valid),
        .in_ready(in_ready), .out_data(out_data), .out_valid(out_valid),
        .out_ready(out_ready)
    );
    always #5 clk=~clk;

    // Drive at falling edge; count transfers BEFORE nonblocking updates.
    task step(input bit r, iv, input [7:0] d, input bit ready,
              input bit want_ready, want_valid, input [7:0] want_data);
        reg take_in, take_out;
        begin
            @(negedge clk);
            rst=r; in_valid=iv; in_data=d; out_ready=ready;
            @(posedge clk);
            if (in_ready !== want_ready) $fatal(1,"in_ready mismatch");
            take_in = !rst && in_valid && in_ready;
            take_out = !rst && out_valid && out_ready;
            if (rst) begin
                // Reset intentionally discards pending samples; new epoch.
                sent=0; received=0;
            end else begin
                // Check output before appending input: forbids empty bypass.
                if (take_out) begin
                    if (received >= sent) $fatal(1,"Duplicate/unexpected output");
                    if (out_data !== expected[received]) $fatal(1,"Order/data mismatch");
                    $display("RECEIVE %0t data=%0d",$time,out_data);
                    received=received+1;
                end
                if (take_in) begin
                    expected[sent]=in_data;
                    sent=sent+1;
                end
            end
            #1;
            if (out_valid !== want_valid || out_data !== want_data)
                $fatal(1,"Post-edge mismatch at %0t: valid=%b data=%0d",
                       $time,out_valid,out_data);
            if ((sent-received) != (out_valid ? 1 : 0))
                $fatal(1,"Occupancy mismatch");
            $display("CHECK %0t push=%b pop=%b valid=%b data=%0d",
                     $time,take_in,take_out,out_valid,out_data);
        end
    endtask

    initial begin
        $timeformat(-9,0," ns",0);
        $dumpfile("runs/learning_ready_valid_buffer.vcd");
        $dumpvars(0,tb_learning_ready_valid_buffer);
        @(posedge clk); #1;
        if (out_valid !== 0 || out_data !== 0) $fatal(1,"Initial reset");
        step(0,1,8,  0, 1,1,8);   // 15 ns: fill empty slot
        step(0,1,12, 0, 0,1,8);   // 25 ns: stalled sender holds 12
        step(0,1,12, 0, 0,1,8);   // 35 ns: still stalled
        step(0,1,12, 1, 1,1,12);  // 45 ns: old 8 out, new 12 in
        step(0,0,99, 1, 1,0,12);  // 55 ns: drain; stale bits remain
        step(0,0,99, 1, 1,0,12);  // empty, no duplicate
        step(0,1,0,  1, 1,1,0);   // valid zero, no empty bypass
        step(0,1,255,1, 1,1,255);
        step(0,1,255,1, 1,1,255); // identical values are distinct samples
        step(0,0,0,  1, 1,0,255);
        if (sent != 5 || received != 5) $fatal(1,"Sequence totals");
        $display("SEQUENCE PASS sent=%0d received=%0d",sent,received);
        step(0,1,42, 0, 1,1,42);
        step(1,1,77, 0, 0,0,0);   // reset overrides full/stall
        step(0,1,9,  1, 1,1,9);
        step(0,0,0,  1, 1,0,9);
        if (sent != 1 || received != 1) $fatal(1,"Reset recovery totals");
        $display("PASS learning_ready_valid_buffer");
        $finish;
    end
endmodule
