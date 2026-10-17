`timescale 1ns/1ps
module tb_learning_line_buffer;
    reg clk=0, rst=1, start=0, clear_done=0, in_valid=0, out_ready=0;
    reg [7:0] in_data=0;
    wire in_ready, out_valid, done_status, has_above, frame_done;
    wire [7:0] current_pixel, above_pixel;
    wire [1:0] out_row, out_col, row, col, state;
    wire [3:0] input_count, output_count;
    integer frame, tx, rx, cycles, j, next_index;
    integer stalls=0, bubbles=0, replacements=0;
    reg pending, pushing, popping, held;
    reg [20:0] saved_payload;
    reg [7:0] saved_cache [0:2];
    learning_line_buffer dut (.*);
    always #5 clk=~clk;
    function automatic [7:0] sample(input integer f, input integer i);
        sample=(f==0) ? (i+1)*10 : 101+i;
    endfunction
    initial begin
        $dumpfile("runs/learning_line_buffer.vcd");
        $dumpvars(0,tb_learning_line_buffer);
        @(posedge clk); #1;
        if (state!==0 || out_valid!==0 || input_count!==0) $fatal(1,"Reset");
        for (frame=0; frame<2; frame=frame+1) begin
            tx=0; rx=0; cycles=0; pending=0;
            @(negedge clk); rst=0; start=1; clear_done=0; in_valid=0; out_ready=0;
            @(posedge clk); #1;
            if (state!==1 || row!==0 || col!==0 || input_count!==0 || out_valid!==0)
                $fatal(1,"Start");
            while(rx<9) begin
                @(negedge clk);
                start=(cycles==3); // Busy start must not restart this frame.
                if (!pending && tx<9 && cycles%4!=1) pending=1;
                in_valid=pending; in_data=sample(frame,tx);
                out_ready=(cycles%5>=2);
                @(posedge clk);
                pushing=in_valid && in_ready; popping=out_valid && out_ready;
                held=out_valid && !out_ready;
                saved_payload={current_pixel,above_pixel,has_above,out_row,out_col};
                for(j=0;j<3;j=j+1) saved_cache[j]=dut.line_buf[j];
                if(held) stalls=stalls+1;
                if(!in_valid) bubbles=bubbles+1;
                if(pushing && popping) replacements=replacements+1;
                if(popping) begin
                    if(rx>=tx || out_row!==(rx/3) || out_col!==(rx%3) ||
                       current_pixel!==sample(frame,rx) || has_above!==(rx>=3))
                        $fatal(1,"Output payload frame=%0d index=%0d",frame,rx);
                    if(above_pixel!==((rx>=3) ? sample(frame,rx-3) : 8'd0))
                        $fatal(1,"Wrong neighbor or previous-frame leakage");
                    $display("OUTPUT frame=%0d pos=(%0d,%0d) current=%0d above=%0d has_above=%b",
                             frame+1,out_row,out_col,current_pixel,above_pixel,has_above);
                    rx=rx+1;
                end
                if(pushing) begin tx=tx+1; pending=0; end
                #1;
                for(j=0;j<3;j=j+1) begin
                    if(pushing && j==(tx-1)%3) begin
                        if(dut.line_buf[j]!==sample(frame,tx-1)) $fatal(1,"Cache write");
                    end else if(dut.line_buf[j]!==saved_cache[j]) $fatal(1,"Cache changed without write");
                end
                if(held && (!out_valid ||
                   {current_pixel,above_pixel,has_above,out_row,out_col}!==saved_payload))
                    $fatal(1,"Backpressure hold");
                next_index=(tx<9) ? tx : 8;
                if(row!==(next_index/3) || col!==(next_index%3)) $fatal(1,"Cursor");
                if(input_count!==tx || output_count!==rx || dut.full!==(tx-rx==1))
                    $fatal(1,"Counts or occupancy");
                if(done_status!==(rx==9) || frame_done!==(popping && rx==9)) $fatal(1,"Completion");
                cycles=cycles+1;
                if(cycles>100) $fatal(1,"Timeout");
            end
            @(negedge clk); start=1; clear_done=1; in_valid=1; in_data=255;
            @(posedge clk);
            if(in_ready || out_valid) $fatal(1,"Extra transfer in DONE");
            #1;
            if(state!==0 || row!==2 || col!==2 || frame_done!==0) $fatal(1,"Clear");
        end
        @(negedge clk); rst=1; start=0; clear_done=0;
        @(posedge clk); #1;
        if(out_valid!==0 || row!==0 || col!==0 || has_above!==0) $fatal(1,"Final reset");
        if(stalls==0 || bubbles==0 || replacements==0) $fatal(1,"Missing coverage");
        $display("PASS line_buffer frames=2 stalls=%0d bubbles=%0d replacements=%0d",
                 stalls,bubbles,replacements);
        $finish;
    end
endmodule
