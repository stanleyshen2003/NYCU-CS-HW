.data
msg1:	.asciiz "Enter the number n = "
msg2:	.asciiz " is a prime"
msg3:	.asciiz " is not a prime, the nearest prime is"
msg4:	.asciiz " "

.text
.globl main
#------------------------- main -----------------------------
main:
# print msg1 on the console interface
		li      $v0, 4			# printf for cin	
		la      $a0, msg1			
		syscall                 	
 
# read the input integer in $v0
 		li      $v0, 5          	# call system call: read integer
  		syscall                 	
  		move    $a1, $v0
  		move	$t6, $v0  
		jal prime
  		
  		
  		bne $v0, $zero, isPrime
  		addi $sp, $sp, -4
  		sw $s0, 0($sp)
# else
		move      $a0, $a1		# a1 = n
		li      $v0, 1			
		syscall 
		li      $v0, 4			
		la      $a0, msg3		
		syscall 
		addi $t7, $zero, 0		# t7 = flag
		addi $s0, $zero, 1		# t0 = i
notPrime: 
  		sub $a1, $t6, $s0
  		jal prime 
  		
  		beq $v0, $zero, secondIf
  		 
  		li      $v0, 4			
		la      $a0, msg4			
		syscall 
  		addi $t7, $t7, 1
  		move $a0, $a1
  		li      $v0, 1			
		syscall 
secondIf:
		add $a1, $t6, $s0
		jal prime
		
		beq $v0, $zero, endSecond
  		li      $v0, 4			
		la      $a0, msg4			
		syscall 
  		addi $t7, $t7, 1
  		move $a0, $a1
  		li      $v0, 1			
		syscall
endSecond:
		bne $zero $t7 finish
		addi $s0, $s0, 1
		j notPrime
		  
isPrime:
		move      $a0, $a1
		li      $v0, 1			# call system call: print string
		syscall 
		li      $v0, 4			# call system call: print string
		la      $a0, msg2			# load address of string into $a0
		syscall  
		
finish: 
		lw $s0, 0($sp)
		addi $sp, $sp,4
		li $v0, 10					# call system call: exit
  		syscall						
 

 
#------------------------- procedure factorial -----------------------------
# load argument n in a0, return value in v0. 
.text
prime:		addi $t0, $a1, -1
		addi $t1, $zero, 2
		bne $t0, $zero, loop
		add $v0, $zero, $zero
		jr $ra
loop:
		mul $t2, $t1, $t1		# i = t1
		slt $t3, $a1, $t2
		bne $t3, $zero, endLoop
		
		div $t4, $a1, $t1
		mfhi $t5
		beq $t5, $zero, out
		
		addi $t1, $t1, 1
		j loop
endLoop:
		addi $v0, $zero, 1 
		jr $ra
out:	
		addi $v0, $zero, 0
		jr $ra
