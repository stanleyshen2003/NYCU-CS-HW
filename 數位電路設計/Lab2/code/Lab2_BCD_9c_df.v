module Lab2_BCD_9c_df (input [3:0] BCD, output [3:0] BCD_9c);

    assign BCD_9c[3] = ~BCD[3] & ~BCD[2] & ~BCD[1];
    assign BCD_9c[2] = BCD[2] ^ BCD[1];
    assign BCD_9c[1] = BCD[1];
    assign BCD_9c[0] = ~BCD[0];    

endmodule
