module Lab2_BCD_9c_beh (input [3:0] BCD, output reg [3:0] BCD_9c);

    always @(*) begin
        BCD_9c[3] = ~BCD[3] & ~BCD[2] & ~BCD[1];
        BCD_9c[2] = BCD[2] ^ BCD[1];
        BCD_9c[1] = BCD[1];
        BCD_9c[0] = ~BCD[0]; 
    end   

endmodule