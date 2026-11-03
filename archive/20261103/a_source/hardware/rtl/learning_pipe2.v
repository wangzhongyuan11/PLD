module learning_pipe2 (
    input wire clk,
    input wire rst,
    input wire en,
    input wire [7:0] d,
    output reg [7:0] q1,
    output reg [7:0] q2
);
    always @(posedge clk) begin
        if (rst) begin
            q1 <= 8'd0;
            q2 <= 8'd0;
        end else if (en) begin
            q1 <= d;
            q2 <= q1;
        end
    end
endmodule
