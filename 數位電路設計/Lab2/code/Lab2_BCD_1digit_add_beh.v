module Lab2_BCD_1digit_add_beh (input [3:0] BCD_X, BCD_Y, input cin, output reg [3:0] BCD_S, output reg cout);
    reg [3:0] Z, temp;
    wire zer, dontcare;
    reg c1, and32, and31;
    wire [3:0] tempZ, tempBCD_S;
    wire tempc1;
    assign zer = 1'b0;
    Lab2_4_bit_CLA_df fullAdder1(BCD_X,BCD_Y, cin, tempZ, tempc1);
    Lab2_4_bit_CLA_df fullAdder2(Z, temp, zer, tempBCD_S, dontcare);

    always @(*) begin
        Z = tempZ;
        c1 = tempc1;
        and32 = Z[3] & Z[2];
        and31 = Z[3] & Z[1];
        cout = and32 | and31 | c1;
        temp[0] = 0;
        temp[1] = cout;
        temp[2] = cout;
        temp[3] = 0;
        BCD_S = tempBCD_S;
    end
  
endmodule
