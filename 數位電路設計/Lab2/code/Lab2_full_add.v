module	Lab2_full_add (input a, b, cin,output sum, cout);

	wire s1, c1, c2;
    Lab2_half_add half1(a, b, s1, c1);
    Lab2_half_add half2(s1, cin, sum, c2);
    or #(2) or1(cout, c2, c1);
endmodule