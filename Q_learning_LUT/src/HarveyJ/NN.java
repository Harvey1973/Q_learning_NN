
package HarveyJ;
import java.util.Arrays;


public class NN {
	public static double  [][] weights_1 = initialize_weights(4,2);
	
	public static double [][] bias_1 = initialize_bias(4,1);
	public static double [][] weights_2 = initialize_weights(1,4);
	public static double [][] bias_2 = initialize_bias(1,1);
	/* momentum matrix*/
	public static double [][] v1 = create_v(weights_1);
	public static double [][] v2 = create_v(weights_2);
	public static double [][] v3 = create_v(bias_1);
	public static double [][] v4 = create_v(bias_2);
	public static double cost = 0.0;

	public static double[][] Matrix_trans(double [][] m){
		int row_num = m.length;
		int col_num = m[0].length;
		double[][] temp = new double[col_num][row_num];
	    for (int i = 0; i < m.length; i++)
	         for (int j = 0; j < m[0].length; j++)
	              temp[j][i] = m[i][j];
		return temp;
	}
	public static double[][] Matrix_sub(double [][]m,double [][]n){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][]sub = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				sub[i][j] = m[i][j] - n[i][j];
			}
		}
		return sub;
	}
	public static double [][] Matrix_dot(double [][]m, double [][]n){
		double [][] result = new double [m.length][n[0].length];
		for (int i = 0; i < m.length; i++) { 
		    for (int j = 0; j < n[0].length; j++) { 
		        for (int k = 0; k < m[0].length; k++) { 
		            result[i][j] += m[i][k] * n[k][j];
		        }
		    }
		}
		return result;
	}
	public static double [][] Matrix_mul(double [][] m, double [][] n){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][]result = new double [row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				result[i][j] = m[i][j] * n[i][j];
			}
		}
		return result;
	}
	public static double [][] initialize_weights(int row,int col){
		double [][]weights = new double[row][col];
		double min = -0.5;
		double max = 0.5;
		
		for(int i=0; i<row; i++) {
			
			for(int j=0; j<col; j++) {
				/*change this to random initialization*/
				weights[i][j] = (double)(Math.random() * (max - min) + min);
			}
		}
		return weights;
	}
	public static double [][] initialize_bias(int row,int col){
		double [][]weights = new double[row][col];
		
		for(int i=0; i<row; i++) {
			
			for(int j=0; j<col; j++) {
				/*change this to random initialization*/
				weights[i][j] = 0.0;
			}
		}
		return weights;
	}
	
	public static double[][] sigmoid(double [][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				/*perform sigmoid activation for every element in the matrix  1/(1+e^(-x))*/
				results[i][j] = 1/(1+Math.exp(-1*m[i][j]));
			}
		}
		return results;
	}
	public static double [][] bipolar_sigmoid(double [][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				/*perform sigmoid activation bipolar version for every element in the matrix  1/(1+e^(-x))*/
				results[i][j] = (1-Math.exp(-1*m[i][j]))/(1+Math.exp(-1*m[i][j]));
			}
		}
		return results;
		
	}
	public static double [][] sigmoid_deriv_bipolar(double [][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		double [][] ones = new double [row_num][col_num];
		ones = initialize_ones(m);
		results = constant_mul(0.5,Matrix_mul(Matrix_add(ones,m),Matrix_sub(ones,m))); 
		return results;
		
	}
	public static double [][] constant_mul(double c,double[][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				results[i][j] = c*m[i][j];
			}
		}
		return results;
	}
	public static double [][] Matrix_add(double [][]m,double [][] n){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] sum = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				sum[i][j] = m[i][j] + n[i][j];
			}
		}
		return sum;
	}
	public static double [][] initialize_ones(double [][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] ones = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				ones[i][j] = 1.0;
			}
		}
		return ones;
		
	}
	public static double [][] sigmoid_deriv(double [][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		double [][] ones = new double [row_num][col_num];
		ones = initialize_ones(m);
		results = Matrix_mul(m,Matrix_sub(ones,m)); 
		return results;
		
	}

	public static double [][] update(double lr, double[][]weights,double[][]d_weights){
		/* return the updated weights matrix*/
		int row_num = weights.length;
		int col_num = weights[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				results[i][j] = weights[i][j]-lr*d_weights[i][j];

			}
		}
		return results;
	}
	public static double [][] lr_product(double lr,double[][]d_weights){
		/* return the updated weights matrix*/
		int row_num = d_weights.length;
		int col_num = d_weights[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				results[i][j] =-lr*d_weights[i][j];

			}
		}
		return results;
	}
 
	public static double[][] create_v(double[][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				results[i][j] =0.0;

			}
		}
		return results;
	}
	public static double[][] create_mu(double[][]m,double mu){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				
				results[i][j] = mu;

			}
		}
		return results;
	}
	public static double[][] update_v(double[][]w,double [][]dw,double mu,double[][]v){
		double [][] mu_matrix = create_mu(w,mu);
		return Matrix_add(Matrix_mul(mu_matrix,v),lr_product(0.2,dw));
		
	}
	public static double [][] update_weights(double[][]w,double[][]v){

		return Matrix_add(w,v);
	}
	public static double compute_cost(double [][]y_predict, double [][] y_train) {
		double cost = 0.0 ;
		for (int i = 0; i< y_predict.length;i++) {
			cost += (y_train[i][0] - y_predict[i][0])*(y_train[i][0] - y_predict[i][0]);
		}
		return cost ;
	}
	public static int train(double[][][]X,double[][][]Y,double[][]weights_1,double[][]bias_1,double[][]weights_2,double[][]bias_2,int num_iterations, double error_threshold,double[][]v1,double[][]v2,double[][]v3,double[][]v4,double mu,double cost) {
		int count = 0;
		int sum = 0;
		/* outer loop for iterations */
		for (int j =0; j<num_iterations;j++) {
			if (cost >= error_threshold) {
			cost = 0.0;
			/* iterating through examples*/
			for (int i = 0; i <X.length;i++) {		
				/* Z1 = W1*X[0]+bias_1*/
				double [][] Z1 = Matrix_add(Matrix_dot(weights_1,X[i]),bias_1);
				double [][] A1 = bipolar_sigmoid(Z1);		
				/* Z2 = W2*A1 + bias_2*/
				double [][] Z2 = Matrix_add(Matrix_dot(weights_2,A1),bias_2);
				double [][] A2 = bipolar_sigmoid(Z2);
				cost += (Y[i][0][0]-A2[0][0])*(Y[i][0][0]-A2[0][0]);
				/*back prop*/
				/* dZ2 = (A2-Y[0])*(1+A2)*(1-A2)*0.5*/
				double [][]sigmoid_deriv_1 = sigmoid_deriv_bipolar(A2);
				double [][]dZ2 = Matrix_mul(Matrix_sub(A2,Y[i]),sigmoid_deriv_1);
				/*dW2 = np.dot(dZ2,A1.T)*/
				double [][]dW2 = Matrix_dot(dZ2,Matrix_trans(A1));
				v2 = update_v(weights_2,dW2,mu,v2);
				weights_2 = update_weights(weights_2,v2);
				/* db2 = dZ2*/
				double [][]db2 = dZ2 ;
				v4 = update_v(bias_2,db2,mu,v4);
				bias_2 = update_weights(bias_2,v4);
				/*dZ1 = np.dot(W2.T,dZ2)*(1+A1)*(1-A1)*0.5*/
				double [][]dZ1 = Matrix_mul(Matrix_dot(Matrix_trans(weights_2),dZ2),sigmoid_deriv_bipolar(A1));
				/*dW1 = np.dot(dZ1,X.T)*/
				double [][]dW1 = Matrix_dot(dZ1,Matrix_trans(X[i]));
				v1 = update_v(weights_1,dW1,mu,v1);
				weights_1 = update_weights(weights_1,v1);
				/*db1 = dZ1*/
				double [][]db1 = dZ1 ;
				v3 = update_v(bias_1,db1,mu,v3);
				bias_1 = update_weights(bias_1,v3);
			}
		cost = cost*0.5;
		count+=1;
		sum = j;
		System.out.println(count + " "+cost);
		}  /* if loop*/

		
		} /* this bracket closes the outer most loop*/
		return sum;
	}
	public static double train_2(double[][]X,double[][]Y,double[][]w1,double[][]b1,double[][]w2,double[][]b2,double[][]v11,double[][]v22,double[][]v33,double[][]v44,double mu) {

		/* stochastic gradient descent , look at one example at a time */
			
	/* Z1 = W1*X[0]+bias_1*/
	double [][] Z1 = Matrix_add(Matrix_dot(w1,X),b1);
	double [][] A1 = sigmoid(Z1);		
	/* Z2 = W2*A1 + bias_2*/
	double [][] Z2 = Matrix_add(Matrix_dot(w2,A1),b2);
	double [][] A2 = sigmoid(Z2);
	cost = compute_cost(A2,Y);
	/*back prop*/
	/* dZ2 = (A2-Y[0])*(A2)*(1-A2)*/
	double [][]sigmoid_deriv_1 = sigmoid_deriv(A2);
	double [][]dZ2 = Matrix_mul(Matrix_sub(A2,Y),sigmoid_deriv_1);
	/*dW2 = np.dot(dZ2,A1.T)*/
	double [][]dW2 = Matrix_dot(dZ2,Matrix_trans(A1));
	v22 = update_v(w2,dW2,mu,v22);
	w2 = update_weights(w2,v22);
	/* db2 = dZ2*/
	double [][]db2 = dZ2 ;
	v44 = update_v(b2,db2,mu,v44);
	b2 = update_weights(b2,v44);
	/*dZ1 = np.dot(W2.T,dZ2)*(A1)*(1-A1)*/
	double [][]dZ1 = Matrix_mul(Matrix_dot(Matrix_trans(w2),dZ2),sigmoid_deriv(A1));
	/*dW1 = np.dot(dZ1,X.T)*/
	double [][]dW1 = Matrix_dot(dZ1,Matrix_trans(X));
	v11 = update_v(w1,dW1,mu,v11);
	w1 = update_weights(w1,v11);

	/*db1 = dZ1*/
	double [][]db1 = dZ1 ;
	v33 = update_v(b1,db1,mu,v33);
	b1 = update_weights(b1,v33);
	System.out.println(Arrays.deepToString(w1));
	weights_1 = w1 ;
	weights_2 = w2;
	bias_1 = b1;
	bias_2 = b2;
	v1 = v11;
	v2 = v22;
	v3=  v33;
	v4 = v44;
	
	return cost;
}
	public static double [][] predict(double[][]X,double [][] weights_1,double [][] bias_1,double[][] weights_2,double [][] bias_2) {
		/* predict */
		double [][] result = null ;
		
		double [][] Z1 = Matrix_add(Matrix_dot(weights_1,X),bias_1);
			
		double [][] A1 = bipolar_sigmoid(Z1);
	
	
		/* Z2 = W2*A1 + bias_2*/
		double [][] Z2 = Matrix_add(Matrix_dot(weights_2,A1),bias_2);
		double [][] A2 = bipolar_sigmoid(Z2);
		
		result = A2 ;
		
		return result;
	}

		

}


