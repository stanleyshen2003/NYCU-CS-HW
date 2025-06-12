module Lab2_BCD_1digit_add_df (input [3:0] BCD_X, BCD_Y, input cin, output [3:0] BCD_S, output cout);

    wire [3:0] Z, temp;
    wire dontcare;
    wire zero;
    wire c1, and32, and31;
    Lab2_4_bit_CLA_df fullAdder1(BCD_X, BCD_Y, cin, Z, c1);
    assign and32 = Z[3] & Z[2];
    assign and31 = Z[3] & Z[1];
    assign cout = and32 | and31 | c1;
    assign temp[0] = 0;
    assign temp[1] = cout;
    assign temp[2] = cout;
    assign temp[3] = 0;
    assign zero = 0;
    Lab2_4_bit_CLA_df fullAdder2(Z, temp, zero, BCD_S, dontcare);

endmodule
