// Teaching module: one registered stage, no downstream backpressure.
// A pixel is sampled at a rising edge when in_valid=1.
// The corresponding gray and out_valid appear after that same edge.
module rgb_to_gray (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire [7:0] r,
    input wire [7:0] g,
    input wire [7:0] b,
    output reg out_valid,
    output reg [7:0] gray
);
    wire [15:0] weighted_r = {8'b0, r} * 16'd77;
    wire [15:0] weighted_g = {8'b0, g} * 16'd150;
    wire [15:0] weighted_b = {8'b0, b} * 16'd29;
    wire [15:0] rounded_sum = weighted_r + weighted_g + weighted_b + 16'd128;

    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            gray <= 8'd0;
        end else begin
            out_valid <= in_valid;
            if (in_valid)
                gray <= rounded_sum[15:8];
        end
    end
endmodule
