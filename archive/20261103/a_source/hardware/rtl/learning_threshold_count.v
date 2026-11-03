module learning_threshold_count (
    input wire clk, rst,
    input wire clear_done,
    input wire [7:0] in_data,
    input wire in_valid,
    output wire in_ready,
    output wire out_mask, out_valid,
    input wire out_ready,
    output reg [16:0] pixel_count, target_count,
    output reg frame_done, done_status
);
    reg full, mask_reg;
    wire push, pop;
    // Fixed threshold; one registered result slot, no data bypass.
    // Reset discards pending results and clears this counting interval.
    assign in_ready = !full || out_ready;
    assign out_valid = full;
    assign out_mask = mask_reg;
    assign push = in_valid && in_ready;
    assign pop = out_valid && out_ready;

    // Same-clock acknowledgement; reset > new completion > acknowledgement.
    // Clearing status does not clear counters or start a new frame.
    always @(posedge clk) begin
        if (rst)
            done_status <= 1'b0;
        else if (pop && pixel_count == 17'd5)
            done_status <= 1'b1;
        else if (clear_done)
            done_status <= 1'b0;
    end

    always @(posedge clk) begin
        if (rst) begin
            full <= 1'b0;
            mask_reg <= 1'b0;
        end else if (push) begin
            mask_reg <= (in_data < 8'd100);
            full <= 1'b1;
        end else if (pop) begin
            full <= 1'b0;
        end
    end

    // Counts completed OUTPUT transfers, not arrivals or occupied cycles.
    // Teaching contract: six inputs per frame, reset before the next frame.
    // No protection against extra inputs or counter overflow yet.
    always @(posedge clk) begin
        if (rst) begin
            pixel_count <= 17'd0;
            target_count <= 17'd0;
            frame_done <= 1'b0;
        end else begin
            frame_done <= 1'b0;
            if (pop) begin
                pixel_count <= pixel_count + 17'd1;
                if (out_mask)
                    target_count <= target_count + 17'd1;
                if (pixel_count == 17'd5)
                    frame_done <= 1'b1;
            end
        end
    end
endmodule
