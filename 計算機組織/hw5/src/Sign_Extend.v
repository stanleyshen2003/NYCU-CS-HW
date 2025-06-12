module Sign_Extend( data_i, data_o );

//I/O ports
input	[16-1:0] data_i;
output	[32-1:0] data_o;

//Internal Signals
wire	[32-1:0] data_o;

//Sign extended
/*your code here*/
assign data_o[16-1:0] = data_i;
genvar i;
for(i = 16; i < 32; i = i + 1)
  begin
    assign data_o[i] = data_i[16-1];
  end
always@(data_i)
begin
	//$display("signoutput:%d", data_o);
end
endmodule      
