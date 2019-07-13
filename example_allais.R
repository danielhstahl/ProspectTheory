choice1_option1=c(1)
choice1_option2=c(0, 1, 5)
choice2_option1=c(0, 1)
choice2_option2=c(0, 5)

p_choice1_option1=c(1)
p_choice1_option2=c(.01, .89, .1)
p_choice2_option1=c(.89, .11)
p_choice2_option2=c(.90, .1)

state_space=expand.grid(choice1_option1, choice1_option2, choice2_option1, choice2_option2)
p_realworld=expand.grid(p_choice1_option1, p_choice1_option2, p_choice2_option1, p_choice2_option2)
p_combined=apply(p_realworld, 1, prod)

compute_expectation=function(state_space, p, func=function(x){x}){
  rbind(p)%*%func(as.matrix(state_space))
}
utility=function(x){log(100000+1000000*x)}
raw_expectation=compute_expectation(state_space, p_combined)
log_utility_expectation_init=compute_expectation(state_space, p_combined, utility)

#q=c(3, 10, 1, 6, 1, 2, 9, 1, 1, 4, 5, 1)
#q_combined=q/sum(q)
q_combined=c(
  0.0393972092,
  0.1599723308, 
  0.0208296422, 
  0.0353572647, 
  0.0718684143, 
  0.1810057193, 
  0.0002355861,
  0.1270205081,
  0.0678170439, 
  0.1805296005,
  0.0382197078,
  0.0777469731
)
q_r=round(q_combined, 2)
q_r[7]=.01
q_r[11]=0.03
q_r[12]=0.07
sum(q_r)

actual_utility_expectation=compute_expectation(state_space, q_combined, utility)
compute_expectation(state_space, q_r, utility)
continue=T
while(continue){
  q_p=runif(length(p_combined))
  q_p=q_p/sum(q_p)
  res=compute_expectation(state_space, q_p, utility)
  if(res[1]>res[2]&&res[3]<res[4]&&res[4]<res[1]){
    continue=F
    print(q_p)
  }
}
