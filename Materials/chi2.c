#include <cpgplot.h>
#include <nrutil.h>
#include <fitsio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define pi 3.14159265358979323846
#define sqrt2 1.4142135623731


#define N 400

static float rc=pi/180.0;

float gaussian(float x,float mu,float sig) {
	return 1./(sqrt(2*pi)*sig)*expf(-0.5*pow((x-mu)/sig,2));
}

int main (int argc, const char * argv[]) {
	FILE *INFILE;
	char copy[2000],line[2000],label[1000],*filename;
	
	float rl[9]={-0.5, 0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0, 1.7};
	float rr[9]={ 0.0, 0.0,  0.0,  0.0,  0.6,  1.0,  1.0, 1.0, 1.0};
	float rg[9]={ 0.0, 0.0,  0.0,  1.0,  1.0,  1.0,  0.6, 0.0, 1.0};
	float rb[9]={ 0.0, 0.3,  0.8,  1.0,  0.3,  0.0,  0.0, 0.0, 1.0};
	
	float hl[7]={0.0, 0.001, 0.2, 0.4, 0.6, 0.99, 1.0};
	float hr[7]={0.0, 0.0,   0.5, 1.0, 1.0, 1.0,  1.0};
	float hg[7]={0.0, 0.0,   0.0, 0.5, 1.0, 1.0,  1.0};
	float hb[7]={0.0, 0.0,   0.0, 0.0, 0.3, 1.0,  1.0};
	
	float gl[2]={0.0, 1.0};
	float gr[2]={0.0, 1.0};
	float gg[2]={0.0, 1.0};
	float gb[2]={0.0, 1.0};
	
	int c1,c2,nc,ie,is,je,js;
	float contra, bright;
	float forgd,backg,tr[6],icol;
	
	float c[3],c_like[3];
		
	float vr_tab[N],d_vr_tab[N],vr_tab_member[N],d_vr_tab_member[N];
	float abs_hist[300],hist[300],hist_member[300];
	float member_tab[N];

	float vr_min = -500;
	float vr_max =  100;
	float d_vr;
	
	int i,j,nb,nb_member,index,vr_dim;
	
	nb=0;
	INFILE = fopen("Cas3_total.short","r");
	
	fgets(line,2000,INFILE);
	while(fgets(line,2000,INFILE)!=NULL) {
		if(line[0]!='#') {
			strcpy(copy,line);
			strtok(copy," ");
			for(i=0;i<6;i++) strtok(NULL," ");
			vr_tab[nb] = atof(strtok(NULL," "));
			d_vr_tab[nb] = atof(strtok(NULL," "));
			member_tab[nb] = atoi(strtok(NULL," "));
			
			nb++;
		}
	}
	fclose(INFILE);

	printf("read in %d objects\n",nb);
	
	cpgbeg(0,"vr_hist.ps/cps",0,0);
	cpgqcir(&c1, &c2);
	nc =(int) fmax(0, c2-c1+1);
	contra=1.0;
	bright=0.5;
	cpgscir(c1,c2);
	cpgctab(rl,rr,rg,rb,9,contra,bright);
	
	float vr_list[5]={20,10,5,2,1};
	int d_vr_list_index_max = 5;
	int d_vr_list_index;
	
	float mu_min = -380;
	float mu_max = -360;
	float d_mu = 0.1;
	int mu_dim = (int) ((mu_max-mu_min+1e-4)/d_mu);
	float sig_min = 7.0;
	float sig_max = 12.0;
	float d_sig=0.1;
	int sig_dim = (int) ((sig_max-sig_min+1e-4)/d_sig);
	
	float *chi2,*likelihood,*likelihood_correct,*log_likelihood,*log_likelihood_correct;
	float chi2_min,Lmax;
	float mu,sig,best_mu,best_sig;
	int i_mu,i_sig;
	
	chi2 = (float*) malloc(mu_dim*sig_dim*sizeof(float));
	likelihood = (float*) malloc(mu_dim*sig_dim*sizeof(float));
	likelihood_correct = (float*) malloc(mu_dim*sig_dim*sizeof(float));
	log_likelihood = (float*) malloc(mu_dim*sig_dim*sizeof(float));
	log_likelihood_correct = (float*) malloc(mu_dim*sig_dim*sizeof(float));

	tr[0] = mu_min-0.5*d_mu;
	tr[1] = d_mu;
	tr[2] = 0.0;
	tr[3] = sig_min-0.5*d_sig;
	tr[4] = 0.0;
	tr[5] = d_sig;
	
	is=1;
	ie=mu_dim;
	js=1;
	je=sig_dim;

	cpgsch(1.2);
	cpgslw(3);

	d_vr=10;
	vr_dim = (int)((vr_max-vr_min+1e-4)/d_vr);

	for(j=0;j<vr_dim;j++) {
		abs_hist[j] = vr_min+(j+0.5)*d_vr;
		hist[j]=0;
		hist_member[j]=0;
	}
	
	nb_member=0;
	for(i=0;i<nb;i++) {
		index = (int)((vr_tab[i]-vr_min)/d_vr);
		hist[index]++;
		if(vr_tab[i]<-300.0) {
			hist_member[index]++;
			vr_tab_member[nb_member]=vr_tab[i];
			d_vr_tab_member[nb_member]=d_vr_tab[i];
			nb_member++;
		}
	}

	cpgsvp(0.1,0.4,0.15,0.9);
	cpgswin(vr_min,vr_max,0,10*d_vr);
	cpghist(nb,vr_tab,vr_min,vr_max,vr_dim,1);
	cpgsci(2);
	cpghist(nb_member,vr_tab_member,vr_min,vr_max,vr_dim,1);
	cpgsci(1);
	cpgbox("BNCST",0.0,0,"BNCST",0.0,0);
	cpglab("v\\dr\\u (km/s)","N","");
	cpgpage();


	
	vr_min = -450;
	vr_max = -300;


	
	for(d_vr_list_index=0;d_vr_list_index<d_vr_list_index_max;d_vr_list_index++) {
		
		d_vr = vr_list[d_vr_list_index];
		vr_dim = (int)((vr_max-vr_min+1e-4)/d_vr);
		
		printf("vr_dim = %3d\n",vr_dim);
		
		for(j=0;j<vr_dim;j++) {
			abs_hist[j] = vr_min+(j+0.5)*d_vr;
			hist[j]=0;
			hist_member[j]=0;
		}
		
		nb_member=0;
		for(i=0;i<nb;i++) {
			index = (int)((vr_tab[i]-vr_min)/d_vr);
			hist[index]++;
			if(vr_tab[i]<-300.0) {
				hist_member[index]++;
				vr_tab_member[nb_member]=vr_tab[i];
				d_vr_tab_member[nb_member]=d_vr_tab[i];
				nb_member++;
			}
		}
		
		chi2_min=1e30;
		for(i_mu=0; i_mu<mu_dim; i_mu++) {
			mu = mu_min+(i_mu+0.5)*d_mu;
			for(i_sig=0; i_sig<sig_dim; i_sig++) {
				sig = sig_min+(i_sig+0.5)*d_sig;
				
				chi2[i_sig*mu_dim+i_mu] = 0.0;
				for(i=0;i<vr_dim;i++)
					chi2[i_sig*mu_dim+i_mu]+=pow(hist_member[i] - nb_member*d_vr*gaussian(abs_hist[i],mu,sig),2);
				if(chi2[i_sig*mu_dim+i_mu]<chi2_min) {
					chi2_min = chi2[i_sig*mu_dim+i_mu];
					best_mu = mu;
					best_sig = sig;
				}
			}
		}
			
		c[0] = chi2_min + 2.3;
		c[1] = chi2_min + 6.18;
		c[2] = chi2_min + 11.8;

		cpgsvp(0.1,0.4,0.15,0.9);
		cpgswin(vr_min,vr_max,0,10*d_vr);
		cpghist(nb,vr_tab,vr_min,vr_max,vr_dim,1);
		cpgsci(2);
		cpghist(nb_member,vr_tab_member,vr_min,vr_max,vr_dim,1);
		cpgsci(1);
		cpgbox("BNCST",0.0,0,"BNCST",0.0,0);
		cpglab("v\\dr\\u (km/s)","N","");
		
		cpgsvp(0.55,0.9,0.15,0.9);
		cpgswin(mu_min,mu_max,sig_min,sig_max);
		
		cpgcont(chi2,mu_dim,sig_dim,is,ie,js,je,c,3,tr);
		cpgpt1(best_mu,best_sig,17);
		cpgbox("BNCST",0.0,0,"BNCST",0.0,0);
		sprintf(label,"Bin size = %1.0f km/s",d_vr);
		cpglab("Mean velocity of Gaussian model \\gm (km/s)","Dispersion of model \\gs (km/s)",label);

		
		cpgpage();
	}
	
	printf("%d members\n",nb_member);
	
	
	Lmax=-1e30;
	for(i_mu=0; i_mu<mu_dim; i_mu++) {
		mu = mu_min+(i_mu+0.5)*d_mu;
		for(i_sig=0; i_sig<sig_dim; i_sig++) {
			sig = sig_min+(i_sig+0.5)*d_sig;
						
			log_likelihood[i_sig*mu_dim+i_mu] = 0.0;
			for(j=0;j<nb_member;j++) {
				log_likelihood[i_sig*mu_dim+i_mu]+=log(gaussian(vr_tab_member[j],mu,sig));
			}
			
			if(log_likelihood[i_sig*mu_dim+i_mu]>Lmax) {
				Lmax = log_likelihood[i_sig*mu_dim+i_mu];
				best_mu = mu;
				best_sig = sig;
			}
		}
	}
	for(i=0;i<mu_dim*sig_dim;i++)
		likelihood[i] = expf(log_likelihood[i]-Lmax);
	
	
	c_like[0] = exp(-0.5*2.3);
	c_like[1] = exp(-0.5*6.18);
	c_like[2] = exp(-0.5*11.8);

	cpgsvp(0.55,0.9,0.15,0.9);
	cpgswin(mu_min,mu_max,sig_min,sig_max);
	
	cpgcont(likelihood,mu_dim,sig_dim,is,ie,js,je,c_like,3,tr);
	cpgpt1(best_mu,best_sig,17);
	cpgbox("BNCST",0.0,0,"BNCST",0.0,0);
	cpglab("Mean velocity of Gaussian model \\gm (km/s)","Dispersion of model \\gs (km/s)","Likelihood analysis");
	cpgpage();
	
	Lmax=-1e30;
	for(i_mu=0; i_mu<mu_dim; i_mu++) {
		mu = mu_min+(i_mu+0.5)*d_mu;
		for(i_sig=0; i_sig<sig_dim; i_sig++) {
			sig = sig_min+(i_sig+0.5)*d_sig;
			
			log_likelihood_correct[i_sig*mu_dim+i_mu] = 0.0;
			for(j=0;j<nb_member;j++) {
				log_likelihood_correct[i_sig*mu_dim+i_mu]+=log(gaussian(vr_tab_member[j],mu,sqrt(pow(sig,2)+pow(d_vr_tab_member[j],2))));
			}
			
			if(log_likelihood_correct[i_sig*mu_dim+i_mu]>Lmax) {
				Lmax = log_likelihood_correct[i_sig*mu_dim+i_mu];
				best_mu = mu;
				best_sig = sig;
			}
		}
	}
	for(i=0;i<mu_dim*sig_dim;i++)
		likelihood_correct[i] = expf(log_likelihood_correct[i]-Lmax);
	
	
	c_like[0] = exp(-0.5*2.3);
	c_like[1] = exp(-0.5*6.18);
	c_like[2] = exp(-0.5*11.8);
	
	cpgsvp(0.55,0.9,0.15,0.9);
	cpgswin(mu_min,mu_max,sig_min,sig_max);
	
	cpgcont(likelihood_correct,mu_dim,sig_dim,is,ie,js,je,c_like,3,tr);
	cpgpt1(best_mu,best_sig,17);
	cpgbox("BNCST",0.0,0,"BNCST",0.0,0);
	cpglab("Mean velocity of Gaussian model \\gm (km/s)","Dispersion of model \\gs (km/s)","Likelihood analysis corrected from measurement uncertainties");

	
	cpgend();
	
	system("ps2pdf vr_hist.ps\n");
	
	
	
	
	return 0;
}

