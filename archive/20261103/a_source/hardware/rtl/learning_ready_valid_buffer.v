module learning_ready_valid_buffer (
    input wire clk, rst,
    input wire [7:0] in_data,
    input wire in_valid,
    output wire in_ready,
    output wire [7:0] out_data,
    output wire out_valid,
    input wire out_ready
);
    reg [7:0] data_reg;
    reg full;
    wire push, pop;
    // One slot, no data bypass. Ignore transfers during synchronous reset.
    assign in_ready = !full || out_ready;
    assign out_valid = full;
    assign out_data = data_reg;
    assign push = in_valid && in_ready;
    assign pop = out_valid && out_ready;

    always @(posedge clk) begin
        if (rst) begin
            full <= 1'b0;
            data_reg <= 8'd0;
        end else if (push) begin
            data_reg <= in_data;
            full <= 1'b1;
        end else if (pop) begin
            full <= 1'b0;
        end
    end
endmodule
