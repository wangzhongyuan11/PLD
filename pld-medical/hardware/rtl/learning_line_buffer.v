// Register-array teaching model: fixed 3x3 grayscale frame, one output per input.
// This is not a declaration of a particular FPGA block-RAM read/write mode.
module learning_line_buffer (
    input wire clk, rst, start, clear_done,
    input wire [7:0] in_data,
    input wire in_valid, out_ready,
    output wire in_ready, out_valid, done_status,
    output reg [7:0] current_pixel, above_pixel,
    output reg has_above,
    output reg [1:0] out_row, out_col,
    output reg [1:0] row, col, state,
    output reg [3:0] input_count, output_count,
    output reg frame_done
);
    localparam WAIT=2'd0, RUN=2'd1, DONE=2'd2;
    reg [7:0] line_buf [0:2];
    reg full;
    wire push = in_valid && in_ready;
    wire pop = out_valid && out_ready;
    assign in_ready = !rst && state==RUN && input_count<9 && (!full || out_ready);
    assign out_valid = !rst && state==RUN && full;
    assign done_status = state==DONE;

    always @(posedge clk) begin
        if (rst) begin
            state<=WAIT; full<=0; row<=0; col<=0;
            out_row<=0; out_col<=0; current_pixel<=0; above_pixel<=0;
            has_above<=0; input_count<=0; output_count<=0; frame_done<=0;
            // Storage need not be cleared: each frame writes its first row.
        end else begin
            frame_done<=0;
            case (state)
                WAIT: if (start) begin
                    state<=RUN; full<=0; row<=0; col<=0;
                    out_row<=0; out_col<=0; current_pixel<=0; above_pixel<=0;
                    has_above<=0; input_count<=0; output_count<=0;
                end
                RUN: begin
                    if (push) begin
                        current_pixel<=in_data;
                        // Zero is a display convention when no neighbor exists.
                        above_pixel<=(row==0) ? 8'd0 : line_buf[col];
                        has_above<=(row!=0);
                        line_buf[col]<=in_data;
                        out_row<=row; out_col<=col; full<=1;
                        input_count<=input_count+1'b1;
                        if (col==2) begin
                            if (row!=2) begin col<=0; row<=row+1'b1; end
                        end else col<=col+1'b1;
                    end else if (pop) full<=0;
                    if (pop) begin
                        output_count<=output_count+1'b1;
                        if (output_count==8) begin state<=DONE; frame_done<=1; end
                    end
                end
                DONE: if (clear_done) state<=WAIT;
                default: begin state<=WAIT; full<=0; end
            endcase
        end
    end
endmodule
