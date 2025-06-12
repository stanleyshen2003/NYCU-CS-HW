module t_Lab3_Converter();
    reg X, Clk, Rst;
    wire Z1, Z2;
    Lab3_Converter_state_diagram state_diagram(X, Clk, Rst, Z1);
    Lab3_Converter_structure structure(X, Clk, Rst, Z2);

    initial begin
        Clk = 0;
        #(9)
        forever begin 
            Clk = ~Clk; 
            #(5);
        end
    end

    integer i, j;

    initial begin
        Rst = 1;
        #(1);
        Rst = 0;
        #(1);
        Rst = 1;
    end
    initial begin
        for(i=3;i<13;i++)  begin
            for(j = 0;j<4;j++) begin
                X = i[j];
                #(10);
            end

        end
        $finish;
    end
    initial begin
		$dumpfile("t_Lab3_Converter.vcd");
		$dumpvars;
	end


endmodule