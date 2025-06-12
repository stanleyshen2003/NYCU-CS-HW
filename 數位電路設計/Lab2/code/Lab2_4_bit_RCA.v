module	Lab2_4_bit_RCA (input [3:0] A, B, input cin, output [3:0] S, output cout);

	wire c1, c2, c3;
    Lab2_full_add full1(A[0], B[0], cin, S[0], c1);
    Lab2_full_add full2(A[1], B[1], c1, S[1], c2);
    Lab2_full_add full3(A[2], B[2], c2, S[2], c3);
    Lab2_full_add full4(A[3], B[3], c3, S[3], cout);

endmodule