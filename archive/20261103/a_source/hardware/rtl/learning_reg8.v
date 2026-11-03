module learning_reg8 (
    input wire clk,
    input wire rst,
    input wire en,
    input wire [7:0] d,
    output reg [7:0] q
);
    always @(posedge clk) begin
        if (rst)
            q <= 8'd0;
        else if (en)
            q <= d;
    end
endmodule
