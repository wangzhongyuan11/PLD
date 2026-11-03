// Fixed 3-column, 2-row teaching image. Coordinates accompany the mask slot.
module learning_image_coordinates (
    input wire clk, rst, start, clear_done,
    input wire [7:0] cfg_threshold,
    output wire [7:0] frame_threshold,
    input wire [7:0] in_data,
    input wire in_valid,
    output wire in_ready,
    output wire out_valid, out_mask,
    input wire out_ready,
    output wire [16:0] input_count, pixel_count, target_count,
    output wire frame_done, done_status,
    output wire [1:0] state,
    output reg in_row, out_row,
    output reg [1:0] in_col, out_col
);
    localparam WAIT = 2'd0;
    wire push = in_valid && in_ready;
    learning_frame_threshold_config #(.FRAME_PIXELS(6)) core (.*);

    always @(posedge clk) begin
        if (rst) begin
            in_row <= 0;
            in_col <= 0;
            out_row <= 0;
            out_col <= 0;
        end else if ((state == WAIT) && start) begin
            in_row <= 0;
            in_col <= 0;
            out_row <= 0;
            out_col <= 0;
        end else if (push) begin
            // Save the current pixel's position while advancing the input cursor.
            out_row <= in_row;
            out_col <= in_col;
            if (in_col == 2) begin
                if (in_row == 0) begin
                    in_col <= 0;
                    in_row <= 1;
                end
            end else begin
                in_col <= in_col + 2'd1;
            end
        end
    end
endmodule
