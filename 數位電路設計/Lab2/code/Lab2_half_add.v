module	Lab2_half_add (input a, b, output sum, cout);
	
    xor #(4) xor1(sum, a, b);
    and #(2) and1(cout, a, b);
endmodule