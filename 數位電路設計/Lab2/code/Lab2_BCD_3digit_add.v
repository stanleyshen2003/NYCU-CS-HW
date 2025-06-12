module Lab2_BCD_3digit_add (input [11:0] BCD_X, BCD_Y, input cin, output [11:0] BCD_S, output cout);
    wire cout1, cout2;
    Lab2_BCD_1digit_add_df add1(BCD_X[3:0], BCD_Y[3:0], cin, BCD_S[3:0], cout1);
    Lab2_BCD_1digit_add_df add2(BCD_X[7:4], BCD_Y[7:4], cout1, BCD_S[7:4], cout2);
    Lab2_BCD_1digit_add_df add3(BCD_X[11:8], BCD_Y[11:8], cout2, BCD_S[11:8], cout);

endmodule