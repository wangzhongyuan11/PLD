module learning_frame_threshold_config #(
    parameter integer FRAME_PIXELS = 6
) (
    input wire clk, rst, start, clear_done,
    input wire [7:0] cfg_threshold,
    output reg [7:0] frame_threshold,
    input wire [7:0] in_data,
    input wire in_valid,
    output wire in_ready,
    output wire out_valid, out_mask,
    input wire out_ready,
    output reg [16:0] input_count, pixel_count, target_count,
    output reg frame_done,
    output wire done_status,
    output reg [1:0] state
);
    localparam WAIT = 2'd0, RUN = 2'd1, DONE = 2'd2;
    reg full, mask_reg;
    wire push, pop;
    // Frame length is an elaboration parameter. Threshold is sampled on start.
    // synthesis translate_off
    initial begin
        if (FRAME_PIXELS < 1 || FRAME_PIXELS > 65536)
            $fatal(1, "FRAME_PIXELS must be in 1..65536");
    end
    // synthesis translate_on
    // Requests are sampled at clock edges. A start edge accepts no data.
    assign in_ready = !rst && (state == RUN) && (input_count < FRAME_PIXELS)
                      && (!full || out_ready);
    assign out_valid = !rst && (state == RUN) && full;
    assign out_mask = mask_reg;
    assign done_status = (state == DONE);
    assign push = in_valid && in_ready;
    assign pop = out_valid && out_ready;

    always @(posedge clk) begin
        if (rst) begin
            state <= WAIT;
            frame_threshold <= 8'd0;
            full <= 1'b0;
            mask_reg <= 1'b0;
            input_count <= 17'd0;
            pixel_count <= 17'd0;
            target_count <= 17'd0;
            frame_done <= 1'b0;
        end else begin
            frame_done <= 1'b0;
            case (state)
                WAIT: if (start) begin
                    state <= RUN;
                    frame_threshold <= cfg_threshold;
                    full <= 1'b0;
                    mask_reg <= 1'b0;
                    input_count <= 17'd0;
                    pixel_count <= 17'd0;
                    target_count <= 17'd0;
                end
                RUN: begin
                    if (push) begin
                        input_count <= input_count + 17'd1;
                        mask_reg <= (in_data < frame_threshold);
                        full <= 1'b1;
                    end else if (pop) begin
                        full <= 1'b0;
                    end
                    if (pop) begin
                        pixel_count <= pixel_count + 17'd1;
                        if (out_mask)
                            target_count <= target_count + 17'd1;
                        if (pixel_count == FRAME_PIXELS - 1) begin
                            state <= DONE;
                            frame_done <= 1'b1;
                        end
                    end
                end
                DONE: if (clear_done) state <= WAIT;
                default: begin
                    state <= WAIT;
                    full <= 1'b0;
                    mask_reg <= 1'b0;
                    input_count <= 17'd0;
                    pixel_count <= 17'd0;
                    target_count <= 17'd0;
                end
            endcase
        end
    end
endmodule
