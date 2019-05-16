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

raw_expectation=compute_expectation(state_space, p_combined)
log_utility_expectation_init=compute_expectation(state_space, p_combined, function(x){log(0.1+x)})

q=c(3, 10, 1, 6, 1, 2, 9, 1, 1, 4, 5, 1)
q_combined=q/sum(q)

actual_utility_expectation=compute_expectation(state_space, q_combined, function(x){log(0.1+x)})


### counterfactual

m=t(cbind(c(-2, .5, .1), c(3, 2, 1)))
b=c(2, 1)
b_2=c(-2, 1)
y=cbind(c(-5, 2))

t(m)%*%y
b%*%y

library(pracma)

rref(cbind(m, b))

library(limSolve)

G=diag(ncol(m))
h=(1:ncol(m))*0+.000001

ldei(m, b, G=G, H=h)
ldei(m, b_2, G=G, H=h)

m_works=t(cbind(c(2, .5, 2), c(3, 2, 1)))
#b_works=c(2, 1)
b_works=c(1.9, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
ldei(m_works, b_works, G=G, H=h)
ldei(m_works, b_works)


##experiment

m2=t(cbind(c(-2, .5, 2), c(3, 2, 1)))
#b_works=c(2, 1)
b2=c(1.1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
ldei(m_works, b_works, G=G, H=h)
ldei(m_works, b_works)



ldei(t(as.matrix(state_space)), c(3, 2, 1, .4), G=diag(12), H=(1:12)*0+.000001)
