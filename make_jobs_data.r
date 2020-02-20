library(tidyverse)
job_org = read.csv('jobs_data/behanghel_processed.csv')
total_reps = 64

for(num_rep in 0:total_reps-1){

  job = job_org
  # make ipw weights
  job = job %>% mutate(ipw = POIDS_PZ_6MOIS / (CVE*mean(job$CVE)+OPP*mean(job$OPP)+CLA*mean(job$CLA)))
  # For Z(t) = I[Employed if given t?]
  # set outcomes to Y(t) = Z(t) - E[Z(t)]
  CVEeff = mean(job$ipw * job$EMPLOI_6MOIS * job$CVE) - mean(job$ipw * job$EMPLOI_6MOIS * job$CLA)
  OPPeff = mean(job$ipw * job$EMPLOI_6MOIS * job$OPP) - mean(job$ipw * job$EMPLOI_6MOIS * job$CLA)
  CLAbase = mean(job$ipw * job$EMPLOI_6MOIS * job$CLA)
  job = job %>% mutate(y=EMPLOI_6MOIS - CVE*CVEeff - OPP*OPPeff - CLAbase)
  # make personalization X
  Xcat = c('temps', 'rsqstat', 'zus',  'College_education', 'One_to_5_years_of_exp_in_the_job',  'Technician', 'Skilled_clerical_worker', 'Skilled_blue_colar', 'Q1', 'Q2', 'Q3', 'Q4')
  for(xc in Xcat){
    job[,xc] = factor(job[,xc])
  }
  Xcts = c()
  Xform = paste(Xcat,collapse=" + ")
  # make treatment variable
  job$t = 1*job$CVE + 2*job$OPP + 3*job$CLA

  trnidx = sample(nrow(job),floor(nrow(job)*2/5))
  jobtst = job[-trnidx,]
  jobtrn = job[trnidx,]

  # confound the train data
  o = 4
  kp = (2/3)/o
  jobtrn$z.keep = ((jobtrn$One_to_5_years_of_exp_in_the_job==1)-mean(jobtrn$One_to_5_years_of_exp_in_the_job==1))/sd(jobtrn$One_to_5_years_of_exp_in_the_job==1) + ((jobtrn$Skilled_blue_colar==1)-mean(jobtrn$Skilled_blue_colar==1))/sd(jobtrn$Skilled_blue_colar==1) +  ((jobtrn$rsqstat=='RS2')-mean(jobtrn$rsqstat=='RS2'))/sd(jobtrn$rsqstat=='RS2')
  jobtrn$p.keep =(
  (jobtrn$z.keep<quantile(jobtrn$z.keep,1/3))*(kp/2+jobtrn$CLA*(1-3*kp/2))+
  (jobtrn$z.keep>=quantile(jobtrn$z.keep,2/3))*(kp/2+jobtrn$CVE*(1-3*kp/2))+
  ((jobtrn$z.keep>=quantile(jobtrn$z.keep,1/3)) & (jobtrn$z.keep<quantile(jobtrn$z.keep,2/3)))*(kp/2+jobtrn$OPP*(1-3*kp/2))
  )
  jobtrn$keep = runif(nrow(jobtrn)) <= jobtrn$p.keep
  jobtrn = jobtrn[jobtrn$keep,]

  train_data = data.frame(temps=jobtrn$temps, rsqstat=jobtrn$rsqstat, zus=jobtrn$zus, College_education=jobtrn$College_education,
    One_to_5_years_of_exp_in_the_job=jobtrn$One_to_5_years_of_exp_in_the_job, Technician=jobtrn$Technician,
    Skilled_clerical_worker=jobtrn$Skilled_clerical_worker, Skilled_blue_colar=jobtrn$Skilled_blue_colar,
    Q1=jobtrn$Q1, Q2=jobtrn$Q2, Q3=jobtrn$Q3, Q4=jobtrn$Q4, t=jobtrn$t, ipw=jobtrn$ipw, y=jobtrn$y)
  write.csv(train_data, sprintf("jobs_data/train_%d.csv", num_rep), row.names=FALSE)

  test_data = data.frame(temps=jobtst$temps, rsqstat=jobtst$rsqstat, zus=jobtst$zus, College_education=jobtst$College_education,
    One_to_5_years_of_exp_in_the_job=jobtst$One_to_5_years_of_exp_in_the_job, Technician=jobtst$Technician,
    Skilled_clerical_worker=jobtst$Skilled_clerical_worker, Skilled_blue_colar=jobtst$Skilled_blue_colar,
    Q1=jobtst$Q1, Q2=jobtst$Q2, Q3=jobtst$Q3, Q4=jobtst$Q4, t=jobtst$t, ipw=jobtst$ipw, y=jobtst$y)
  write.csv(test_data, sprintf("jobs_data/test_%d.csv", num_rep), row.names=FALSE)

}

