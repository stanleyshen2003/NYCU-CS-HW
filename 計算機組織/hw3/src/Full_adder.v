module Full_adder(sum, carryOut, carryIn, input1, input2);

	input carryIn, input1, input2;

	output sum, carryOut;

	wire w1, w2, w3;

	xor x1(w1, input1, input2);
	xor x2(sum, w1, carryIn);
	and a1(w2, input1, input2);
	and a2(w3, w1, carryIn);
	or o1(carryOut, w2, w3);

endmodule