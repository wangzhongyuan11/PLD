module learning_pipe2_en_valid (
    input wire clk, rst, en,
    input wire [7:0] d,
    input wire in_valid,
    output reg [7:0] q1, q2,
    output reg v1, v2
);
    always @(posedge clk) begin
        if (rst) begin
            q1 <= 8'd0;
            q2 <= 8'd0;
            v1 <= 1'b0;
            v2 <= 1'b0;
        end else if (en) begin
            q1 <= d;
            q2 <= q1;
            v1 <= in_valid;
            v2 <= v1;
        end
    end
endmodule
