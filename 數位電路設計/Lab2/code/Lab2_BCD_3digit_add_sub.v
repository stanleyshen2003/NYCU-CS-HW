module Lab2_BCD_3digit_add_sub (input [11:0] BCD_X, BCD_Y, input mode, output [11:0] BCD_R, output kout);
    wire cout1, cout2,cout3, cout4, cout5, cout;
    wire [3:0] temp1, temp2, temp3, input1, input2, input3, temp4, temp5,temp6, zero,one;
    wire [11:0] tempR, tempR2;

    Lab2_BCD_9c_df c1(BCD_Y[3:0], temp1);
    Lab2_BCD_9c_df c2(BCD_Y[7:4], temp2);
    Lab2_BCD_9c_df c3(BCD_Y[11:8], temp3);
    
    assign input1 = mode ? temp1 : BCD_Y[3:0];
    assign input2 = mode ? temp2 : BCD_Y[7:4];
    assign input3 = mode ? temp3 : BCD_Y[11:8];

    Lab2_BCD_1digit_add_df add1(BCD_X[3:0], input1, mode, tempR[3:0], cout1);
    Lab2_BCD_1digit_add_df add2(BCD_X[7:4], input2, cout1, tempR[7:4], cout2);
    Lab2_BCD_1digit_add_df add3(BCD_X[11:8], input3, cout2, tempR[11:8], cout);
    assign kout = mode ? ~cout : cout;

    Lab2_BCD_9c_df c4(tempR[3:0], temp4);
    Lab2_BCD_9c_df c5(tempR[7:4], temp5);
    Lab2_BCD_9c_df c6(tempR[11:8], temp6);
    assign zero = 4'b0000;
    assign one = 4'b0001;
    Lab2_BCD_1digit_add_df add4(temp4, one, zero[1], tempR2[3:0], cout3);
    Lab2_BCD_1digit_add_df add5(temp5, zero, cout3, tempR2[7:4], cout4);
    Lab2_BCD_1digit_add_df add6(temp6, zero, cout4, tempR2[11:8], cout5);
    assign BCD_R = mode && kout ? tempR2 : tempR;
endmodule