package HarveyJ;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

import HarveyJ.Q_learning_LUT.Enemy;
import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.RobocodeFileOutputStream;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class Qlearning_NN extends AdvancedRobot {
	public double PI = Math.PI;
	// state representations will be a 4 tuple{x_a,y_a,distance_to_enemy,Bearing}
	// below is the quantized states
	public static int X_coor = 8;
	public static int Y_coor = 6;
	public static int Distance = 10;
	public static int Bearing = 4;
	// below is the unquantizied states feed into neural network




	public static double [][] weights_1 = new double[35][5];
	public static double [][] bias_1 = new double [35][1];
	public static double [][] weights_2 = new double [15][35];;
	public static double [][] bias_2 = new double [15][1];;
	public static double [][] weights_3 = new double [1][15];;
	public static double [][] bias_3 = new double [1][1];;

	/* momentum matrix*/
	public static double [][] v1 = create_v(weights_1);
	public static double [][] v2 = create_v(weights_2);
	public static double [][] v5 = create_v(weights_3);
	public static double [][] v3 = create_v(bias_1);
	public static double [][] v4 = create_v(bias_2);
	public static double [][] v6 = create_v(bias_3);
	public static double mu = 0.95;  //0.95 3 layer works ok
	public static double learning_rate = 0.0003; // 0.00009 3 layer works ok 
	public static double cost = 0.0;
	public static double num_examples = 0;
	static int row_num = 8*6*10*4;
	static int col_num = 5;
	public static double [][] Q_table = new double [row_num][col_num];// This Q_table is a String matrix, use this to save Q_table on disk 
	static double[][] X_train = new double[5][1];
	static double[][] X_train_next = new double [5][1];
	static double[][] X_train_previous = new double [5][1];
	static double[][] Y_train = new double[1][1];
	public static double [][] q_value_actions = new double [col_num][1]; 
	public static double [][] q_value_actions_previous = new double [col_num][1]; 
	public static double [][] q_value_actions_next = new double [col_num][1];
	double[][] Y_out = new double [5][1];
	
	int num_iterations = 1;
	
	//
	double getBearing ;
	public int current_state ;
	public int next_state ;
	private int previousState;
	private int previousAction;
	
	static double current_q = 0.0;
	static double next_q = 0.0;
	
	double reward = 0.0;
	double total_reward_per_action = 0.0;
	double cum_reward;
	public static double [] win_rate = new double [300];
	public static double []error = new double[2500];
	public static int win_count = 0;
	public static int index_win = 0;
	// hyper paramaters 
	double alpha = 0.1;  // learning rate - 0.15
	double gamma = 0.9;  // discount factor
	double epsilon = 0.0;
	

	boolean initialize = false;
	boolean learning = true ;
	boolean offpolicy = true;
	boolean synchrnous = true;
	boolean quantized = false;
	
	double [][] Q_table_double = new double[row_num][col_num]; // This Q_table is a double matrix , use this to perform numeric operations 
	public static int index1 = 0;
	public static int iter = 0;
	public static int count = 0;

	public static Enemy enemy ;
	
	public class Enemy{
		public int y;
		public int x;
		public int distance;
		public int bearing;
		public double x_2;
		public double y_2;
		public double distance_2;
		public double bearing_2;
	}
	
	public void initialize_Q_table(){
		for (int i=0; i<row_num; i++)
		for (int j=0; j<col_num; j++)
		Q_table[i][j]=0.0;
	}


	public void saveTable(File file){
		PrintStream w = null;
		try {
			w = new PrintStream(new RobocodeFileOutputStream(file));
			for (int i = 0; i < Q_table.length; i++)
				for (int j = 0; j < col_num; j++)
					w.println(new Double(Q_table[i][j]));
			if (w.checkError())
				System.out.println("Could not save the data!");
			w.close();
		}
		catch (IOException e) {
			System.out.println("IOException trying to write: " + e);
		}
		finally {
			try {
				if (w != null)
					w.close();
				}
			catch (Exception e) {
				System.out.println("Exception trying to close witer: " + e);
			}
		}
	}// working 
	public void saveWeights(File file,double [][] weights){
		PrintStream w = null;
		try {
			w = new PrintStream(new RobocodeFileOutputStream(file));
			for (int i = 0; i < weights.length; i++)
				for (int j = 0; j < weights[0].length; j++)
					w.println(new Double(weights[i][j]));
			System.out.println("Saving weights");
			if (w.checkError())
				System.out.println("Could not save the weights!");
			w.close();
		}
		catch (IOException e) {
			System.out.println("IOException trying to write: " + e);
		}
		finally {
			try {
				if (w != null)
					w.close();
				}
			catch (Exception e) {
				System.out.println("Exception trying to close witer: " + e);
			}
		}
	}// working 
	public void saveBias(File file,double [][] bias){
		PrintStream w = null;
		try {
			w = new PrintStream(new RobocodeFileOutputStream(file));
			for (int i = 0; i < bias.length; i++)
				for (int j = 0; j < bias[0].length; j++)
					w.println(new Double(bias[i][j]));
			if (w.checkError())
				System.out.println("Could not save the weights!");
			w.close();
		}
		catch (IOException e) {
			System.out.println("IOException trying to write: " + e);
		}
		finally {
			try {
				if (w != null)
					w.close();
				}
			catch (Exception e) {
				System.out.println("Exception trying to close witer: " + e);
			}
		}
	}// working 
	public void saveData(File file){
			PrintStream w = null;
			try {
				w = new PrintStream(new RobocodeFileOutputStream(file));
				for (int i = 0; i < win_rate.length; i++)
					w.println(new Double(win_rate[i]));
			if (w.checkError())
				System.out.println("Could not save the data!");
				w.close();
			}
			catch (IOException e) {
				System.out.println("IOException trying to write: " + e);
			}
			finally {
				try {
					if (w != null)
						w.close();
			}
			catch (Exception e) {
			System.out.println("Exception trying to close witer: " + e);
				}
			}
	}
	public void saveError(File file){
		PrintStream w = null;
		try {
			w = new PrintStream(new RobocodeFileOutputStream(file));
			for (int i = 0; i < error.length; i++)
				w.println(new Double(error[i]));
		if (w.checkError())
			System.out.println("Could not save the data!");
			w.close();
		}
		catch (IOException e) {
			System.out.println("IOException trying to write: " + e);
		}
		finally {
			try {
				if (w != null)
					w.close();
		}
		catch (Exception e) {
		System.out.println("Exception trying to close witer: " + e);
			}
		}
}
public void onRoundEnded(RoundEndedEvent e_1) {			
	System.out.println("win rate is  !!!" + win_count/((double)getRoundNum()+1.0));
	if(getRoundNum() % 10 ==0) {
		win_rate[index_win] = win_count/((double)getRoundNum()+1);
		System.out.println("win rate array is" + Arrays.toString(win_rate));	
		index_win +=1;
		saveData(getDataFile("win_rate.txt"));
	}
	index1=index1+1;
	error[count] = next_q - current_q;
	count+=1;
	saveError(getDataFile("error.txt"));
	System.out.println("The error for this round is"  + error[count]);

	   }

	
	public void loadData(File file){
		BufferedReader r = null;
		try{
			r = new BufferedReader(new FileReader(file));
			for (int i = 0; i < row_num; i++)
				for (int j = 0; j < col_num; j++)
					Q_table[i][j] = Double.parseDouble(r.readLine());
		}
		catch (IOException e) {
			System.out.println("IOException trying to open reader: " + e);
			initialize_Q_table();
			saveTable(getDataFile("LUT.txt"));
		}
			catch (NumberFormatException e) {
				initialize_Q_table();
				saveTable(getDataFile("LUT.txt"));
		}
		catch (NullPointerException e) {
			initialize_Q_table();
			saveTable(getDataFile("LUT.txt"));
	}
		finally {
			try {
				if (r != null)
					r.close();
			}
			catch (IOException e) {
				System.out.println("IOException trying to close reader: " + e);
			}
		}
	} // working oad function works
	
	public void run() {
		enemy= new Enemy();
		// save initial weights

		if( iter == 0) {
			
			weights_1 = initialize_weights(35,5);
			bias_1 = initialize_bias(35,1);
			weights_2 = initialize_weights(15,35);
			bias_2 = initialize_bias(15,1);
			weights_3 = initialize_weights(1,15);
			bias_3 = initialize_bias(1,1);

			saveWeights(getDataFile("weights_1.txt"),weights_1);
			saveWeights(getDataFile("weights_2.txt"),weights_2);
			saveWeights(getDataFile("weights_3.txt"),weights_3);
			saveBias(getDataFile("bias_1.txt"),bias_1);
			saveBias(getDataFile("bias_2.txt"),bias_2);
			saveBias(getDataFile("bias_3.txt"),bias_3);
			
			weights_1 = loadWeights(getDataFile("weights_1.txt"),weights_1);
			weights_2 = loadWeights(getDataFile("weights_2.txt"),weights_2);
			weights_3 = loadWeights(getDataFile("weights_3.txt"),weights_3);
			bias_1 = loadWeights(getDataFile("bias_1.txt"),bias_1);
			bias_2 = loadWeights(getDataFile("bias_2.txt"),bias_2);
			bias_3 = loadWeights(getDataFile("bias_3.txt"),bias_3);
			System.out.println("The loaded weights_1 is" + Arrays.deepToString(weights_1));
			loadData(getDataFile("LUT.txt"));
			iter+=1;
		}
		//turnGunRight(360); // initial scan 
		while(true) {
			if (learning == false) {
				System.out.println("Learning Mode : False");
				X_train[0][0] = enemy.x;
				X_train[1][0] = enemy.y;
				X_train[2][0] = enemy.distance;
				X_train[3][0] = enemy.bearing ;
				for (int j = 0; j< col_num; j++ ) {

					X_train[4][0] = j;
					q_value_actions[j][0] = predict_3(X_train, weights_1,bias_1,weights_2,bias_2, weights_3, bias_3);
					
				}
				System.out.println("The q_values for current state action pair are" + Arrays.deepToString(q_value_actions));
				int action = choose_action(q_value_actions);
				//int action = randInt(0,4);
				take_action(action);
				turnGunRight(360);
			}
			else {
				
				
				if(quantized) {
					turnGunRight(360); // initial scan
					
					X_train[0][0] = enemy.x;
					X_train[1][0] = enemy.y;
					X_train[2][0] = enemy.distance;
					X_train[3][0] = enemy.bearing ;
					
					System.out.println("The current X pos is " + X_train[0][0]);
					for (int j = 0; j< col_num; j++ ) {
	
						X_train[4][0] = j;
						q_value_actions[j][0] = predict_3(X_train, weights_1,bias_1,weights_2,bias_2, weights_3, bias_3);
						
					}
					
					System.out.println("The q_values for current state action pair are" + Arrays.deepToString(q_value_actions));
					int action = choose_action(q_value_actions);
					X_train[4][0] = action;
					current_q = q_value_actions[action][0];
					System.out.println("current_q is " + current_q);
					
					total_reward_per_action = 0.0;
					take_action(action);
					turnGunRight(360);
					
					X_train_next[0][0] = enemy.x;
					X_train_next[1][0] = enemy.y;
					X_train_next[2][0] = enemy.distance;
					X_train_next[3][0] = enemy.bearing ;
					System.out.println("The next X pos is " + X_train_next[0][0]);
					for(int k =0; k< col_num;k++) {
	
					X_train_next[4][0] = k ;
					q_value_actions_next[k][0] = predict_3(X_train_next,weights_1,bias_1,weights_2,bias_2, weights_3, bias_3);
					}
					next_q = max_2(q_value_actions_next);
					System.out.println("NEXT Q is " + next_q);
					System.out.println("Reward is" +total_reward_per_action	);
					current_q += alpha*(total_reward_per_action+gamma*next_q-current_q);
					System.out.println("Desired current_q is " + current_q);
					Y_train[0][0] = current_q;
					
					
					//Y_train[0][0] = total_reward_per_action+gamma*next_q; // working
					cost =train_3(X_train,Y_train,weights_1,bias_1,weights_2,bias_2,weights_3,bias_3,v1,v2,v3,v4,v5,v6,mu);
				
			
		}
				else {
					
					turnGunRight(360); // initial scan
					
					X_train[0][0] = enemy.x_2;
					X_train[1][0] = enemy.y_2;
					X_train[2][0] = enemy.distance_2;
					X_train[3][0] = enemy.bearing_2 ;
					
					System.out.println("The current X pos is " + X_train[0][0]);
					for (int j = 0; j< col_num; j++ ) {

						X_train[4][0] = j;
						q_value_actions[j][0] = predict_3(X_train, weights_1,bias_1,weights_2,bias_2, weights_3, bias_3);
						
					}
					
					System.out.println("The q_values for current state action pair are" + Arrays.deepToString(q_value_actions));
					int action = choose_action(q_value_actions);
					X_train[4][0] = action;
					current_q = q_value_actions[action][0];
					System.out.println("current_q is " + current_q);
					
					total_reward_per_action = 0.0;
					take_action(action);
					turnGunRight(360);
					
					X_train_next[0][0] = enemy.x_2;
					X_train_next[1][0] = enemy.y_2;
					X_train_next[2][0] = enemy.distance_2;
					X_train_next[3][0] = enemy.bearing_2 ;
					System.out.println("The next X pos is " + X_train_next[0][0]);
					for(int k =0; k< col_num;k++) {

					X_train_next[4][0] = k ;
					q_value_actions_next[k][0] = predict_3(X_train_next,weights_1,bias_1,weights_2,bias_2, weights_3, bias_3);
					}
					next_q = max_2(q_value_actions_next);
					
					
					System.out.println("NEXT Q is " + next_q);
					System.out.println("Reward is" +total_reward_per_action	);
					
					current_q += alpha*(total_reward_per_action+gamma*next_q-current_q);
					
					System.out.println("Desired current_q is " + current_q);
					Y_train[0][0] = current_q;
					
					
					cost =train_3(X_train,Y_train,weights_1,bias_1,weights_2,bias_2,weights_3,bias_3,v1,v2,v3,v4,v5,v6,mu);
					
				}

			}

		}
		
	}

	public void onScannedRobot(ScannedRobotEvent e) {
		
		if(quantized) {
			enemy.x = quantize_position(getX());
			enemy.y = quantize_position(getY());
			enemy.distance = quantize_distance(e.getDistance());
			enemy.bearing = quantize_bearing(e.getBearing());
		}
		else {
			enemy.x_2 = getX();
			enemy.y_2 = getY();
			enemy.distance_2 = e.getDistance();
			enemy.bearing_2 = e.getBearing();
	
		}
		fire(3);


		
}
	public void onWin(WinEvent event) {
		System.out.println("winning !!!");
		win_count += 1;
		System.out.println("win count is " + win_count);
		saveTable(getDataFile("LUT.txt"));
		
		saveWeights(getDataFile("weights_1.txt"),weights_1);
		saveWeights(getDataFile("weights_2.txt"),weights_2);
		saveWeights(getDataFile("weights_3.txt"),weights_3);
		saveBias(getDataFile("bias_1.txt"),bias_1);
		saveBias(getDataFile("bias_2.txt"),bias_2);
		saveBias(getDataFile("bias_3.txt"),bias_3);
		
		
		
	}
	public void onDeath(DeathEvent event){
		saveTable(getDataFile("LUT.txt"));
		saveWeights(getDataFile("weights_1.txt"),weights_1);
		saveWeights(getDataFile("weights_2.txt"),weights_2);
		saveWeights(getDataFile("weights_3.txt"),weights_3);
		saveBias(getDataFile("bias_1.txt"),bias_1);
		saveBias(getDataFile("bias_2.txt"),bias_2);
		saveBias(getDataFile("bias_3.txt"),bias_3);
	}
	
	// get the current state
	public int getState(){

		return index(enemy.x,enemy.y,enemy.distance,enemy.bearing);
	}
	


	
	/* used in exlpore mode to randomly pick  actions  */
	public int randInt(int min, int max) {


	    Random rand = new Random();

	    // nextInt is normally exclusive of the top value,
	    // so add 1 to make it inclusive
	    int randomNum = rand.nextInt((max - min) + 1) + min;

	    return randomNum;
	}

	public static int state_index(String state,String[][] Q_table){
		int index = 0;
		for (int i=0;i<Q_table.length;i++) {

			if(Q_table[i][0].equals(state)) {
				index = i;
				
			}
		}
		return index;
	}
	// takes input of an array of forward pass for the current state, input is the array of Q values
	public  int choose_action(double [][] y_out){
		if(Math.random()>epsilon) {
			System.out.println("I am taking greedy action");
			return  argmax_2(y_out);
		}
		else  {
			System.out.println("I am taking random action");
			return  randInt(0,4);
		}
		
	}
	public  double max(double[] array) {
		double largest = -99999999999.0;
		// starts with index 1 because the first element is the state number 
		for ( int i = 0; i < array.length; i++ )
		{
		    if ( array[i] > largest )
		    {
		        largest = array[i];
		        
		    }
		}
		return largest;
	}
	public  double max_2(double[][] array) {
		double largest = -99999999999.0;
		// starts with index 1 because the first element is the state number 
		for ( int i = 0; i < col_num; i++ )
		{
		    if ( array[i][0] > largest )
		    {
		        largest = array[i][0];
		        
		    }
		}
		return largest;
	}
	public  int argmax(int state) {
		int index = 0;   // if the row is all zeros then just take the first action 
		double largest = -99999999;
		// starts with index 1 because the first element is the state number ,but we are only 
		// interested in the actions 
		if (all_zero(Q_table[state])){
			System.out.println("I am taking random action");
			index = randInt(0,4);
		}
		else {
			System.out.println("I am taking greedy action");
		for ( int i = 0; i < Q_table[state].length; i++ )
		{
		    if ( Q_table[state][i] > largest )
		    {
		        largest = Q_table[state][i];
		        index = i;
		    }
		}
		
		}
		return index ;
		
	}
	public  int argmax_2(double [][] y_out) {
		int index = 0;   // if the row is all zeros then just take the first action 
		double largest = -99999999;

		for ( int i = 0; i < col_num ; i++ )
		{
		    if ( y_out[i][0] > largest )
		    {
		        largest = y_out[i][0];
		        index = i;
		    }
		}
		System.out.println("I am taking greedy action index		" + index);
		
		return index ;
		
	}
	public boolean all_zero(double array[]) {
		boolean all0 = true;
		for (int i = 0 ; i< array.length;i++) {
			if(array[i]!=0) {
				all0 = false;
			}
		}
		return all0;
	}
	public void take_action(int action_index) {
		if (synchrnous) {
		if(action_index ==0) {
			ahead(100);

		}
		else if (action_index==1) {
			back(100);
		}
		else if(action_index==2) {
			turnLeft(90);
			ahead(100);
		}
		else if(action_index==3) {
			turnRight(90);
			ahead(100);
		}
		else if(action_index == 4) {
			turnLeft(180);
			ahead(100);
		}
		}
		else {
			if(action_index ==0) {
				setAhead(100);

			}
			else if (action_index==1) {
				setBack(100);
			}
			else if(action_index==2) {
				setTurnLeft(90); 
				setAhead(100);	
			}
			else if(action_index==3) {
				setTurnRight(90);
				setAhead(100);
			}
			else if(action_index == 4) {
				setTurnLeft(180);
				setAhead(100);
			}
			
		}
	} // take action works
	public int quantize_position(double x_coor) {
		int quantized = 0;
		//System.out.println("The argument I receive is "+ x_coor);
		if((x_coor>=0)&&(x_coor<100)) {
			quantized = 0;
		}
		else if((x_coor>=100)&&(x_coor<200)) {
			quantized = 1 ;
		}
		else if((x_coor>=200)&&(x_coor<300)) {
			quantized = 2 ;
		}
		else if((x_coor>=300)&&(x_coor<400)) {
			quantized = 3 ;
		}
		else if((x_coor>=400)&&(x_coor<500)) {
			quantized = 4 ;
		}
		else if((x_coor>=500)&&(x_coor<600)) {
			quantized = 5 ;
		}
		else if((x_coor>=600)&&(x_coor<700)) {
			quantized = 6 ;
		}
		else if((x_coor>=700)&&(x_coor<800)) {
			quantized = 7 ;
		}

		return quantized;
	} // quantized coordinates works
	
	public int quantize_bearing(double bearing_angle) {
		int bearing = 0;
		if ((bearing_angle>=0)&&(bearing_angle<90)){
			bearing= 0;
			
		}
		else if ((bearing_angle>=90)&&(bearing_angle<=180)){
			bearing= 1;
			
		}
		else if ((bearing_angle<0)&&(bearing_angle>=-90)){
			bearing= 2;
			
		}
		else if ((bearing_angle<-90)&&(bearing_angle>=-180)){
			bearing= 3;
			
		}
		return bearing;
		
	}// quantized bearing works
	public int quantize_distance(double distance) {
		int dist = 0;
		if((distance>=0)&&(distance<100)) {
			dist = 0;
		}
		else if((distance>=100)&&(distance<200)) {
			dist = 1 ;
		}
		else if((distance>=200)&&(distance<300)) {
			dist = 2 ;
		}
		else if((distance>=300)&&(distance<400)) {
			dist = 3 ;
		}
		else if((distance>=400)&&(distance<500)) {
			dist = 4 ;
		}
		else if((distance>=500)&&(distance<600)) {
			dist = 5 ;
		}
		else if((distance>=600)&&(distance<700)) {
			dist = 6 ;
		}
		else if((distance>=700)&&(distance<800)) {
			dist = 7 ;
		}
		else if((distance>=800)&&(distance<900)) {
			dist = 8 ;
		}
		else if((distance>=900)&&(distance<1000)) {
			dist = 9 ;
		}
		return dist;
	}
	// reward functions 
	public void onHitRobot(HitRobotEvent event){
		double reward_1 =-2;
		total_reward_per_action += reward_1;
		//System.out.println("HITT robot");
		} 
	public void onBulletHit(BulletHitEvent event){
		double reward_2=3;
		total_reward_per_action += reward_2;
		//System.out.println("HITTING BULLET");
		} 
	public void onHitByBullet(HitByBulletEvent event){
		double reward_3=-3;
		total_reward_per_action += reward_3;
		//System.out.println(" GETTING HIT BY BULLET ");
		}


	public static double predict_3(double[][]X, double [][] weights_1,double [][] bias_1,double[][] weights_2,double [][] bias_2,double[][] weights_3, double[][] bias_3) {
		/* predict */
		
		double [][] Z1 = Matrix_add(Matrix_dot(weights_1,X),bias_1);
			
		double [][] A1 = relu(Z1);
	
	
		/* Z2 = W2*A1 + bias_2*/
		double [][] Z2 = Matrix_add(Matrix_dot(weights_2,A1),bias_2);
		//double [][] A2 = sigmoid(Z2);
		double [][] A2 = relu(Z2);
		double [][] Z3 = Matrix_add(Matrix_dot(weights_3,A2),bias_3);
		//System.out.println(Arrays.deepToString(Z3));
		return Z3[0][0];
		
	}
	public static double compute_cost(double [][]y_predict, double [][] y_train) {
		double cost = 0.0 ;
		for (int i = 0; i< y_predict.length;i++) {
			cost += (y_train[i][0] - y_predict[i][0])*(y_train[i][0] - y_predict[i][0]);
		}
		return cost ;
	}
	public static double train_2(double[][]X,double[][]Y,double[][]w1,double[][]b1,double[][]w2,double[][]b2,double[][]v11,double[][]v22,double[][]v33,double[][]v44,double mu) {

			/* stochastic gradient descent , look at one example at a time */
				
		/* Z1 = W1*X[0]+bias_1*/
		double [][] Z1 = Matrix_add(Matrix_dot(w1,X),b1);
		//double [][] A1 = bipolar_sigmoid(Z1);		
		double [][] A1 = relu(Z1);		
		/* Z2 = W2*A1 + bias_2*/
		double [][] Z2 = Matrix_add(Matrix_dot(w2,A1),b2);
		//double [][] A2 = bipolar_sigmoid(Z2);
		double [][] A2 = Z2;
		//System.out.println("The output of forward pass is  " + Arrays.deepToString(A2));
		cost = compute_cost(A2,Y);
		/*back prop*/
		/* dZ2 = (A2-Y[0])*(A2)*(1-A2)*/
		//double [][]sigmoid_deriv_1 = sigmoid_deriv_bipolar(A2);
		
		//double [][]dZ2 = Matrix_mul(Matrix_sub(A2,Y),sigmoid_deriv_1);
		double [][]dZ2 = Matrix_sub(A2,Y);
		//System.out.println("The groudtruth value is   " + Arrays.deepToString(Y));
		/*dW2 = np.dot(dZ2,A1.T)*/
		double [][]dW2 = Matrix_dot(dZ2,Matrix_trans(A1));
		//v22 = update_v(w2,dW2,mu,v22);
		//w2 = update_weights(w2,v22);
		/* db2 = dZ2*/
		double [][]db2 = dZ2 ;
		//v44 = update_v(b2,db2,mu,v44);
		//b2 = update_weights(b2,v44);
		/*dZ1 = np.dot(W2.T,dZ2)*(A1)*(1-A1)*/
		double [][]dZ1 = Matrix_mul(Matrix_dot(Matrix_trans(w2),dZ2),relu_deriv(A1));
		/*dW1 = np.dot(dZ1,X.T)*/
		double [][]dW1 = Matrix_dot(dZ1,Matrix_trans(X));
		//v11 = update_v(w1,dW1,mu,v11);
		//w1 = update_weights(w1,v11);

		/*db1 = dZ1*/
		double [][]db1 = dZ1 ;
		v22 = update_v(w2,dW2,mu,v22);
		w2 = update_weights(w2,v22);
		v44 = update_v(b2,db2,mu,v44);
		b2 = update_weights(b2,v44);
		v33 = update_v(b1,db1,mu,v33);
		b1 = update_weights(b1,v33);
		v11 = update_v(w1,dW1,mu,v11);
		w1 = update_weights(w1,v11);
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
	public static double train_3(double[][]X,double[][]Y,double[][]w1,double[][]b1,double[][]w2,double[][]b2,double w3[][],double b3[][],double[][]v11,double[][]v22,double[][]v33,double[][]v44,double[][] v55,double[][] v66,double mu) {

		/* stochastic gradient descent , look at one example at a time */
				
		/* Z1 = W1*X[0]+bias_1*/
		double [][] Z1 = Matrix_add(Matrix_dot(w1,X),b1);
		//double [][] A1 = bipolar_sigmoid(Z1);		
		double [][] A1 = relu(Z1);		
		/* Z2 = W2*A1 + bias_2*/
		double [][] Z2 = Matrix_add(Matrix_dot(w2,A1),b2);
	    /* A2 = relu(Z2)*/
		double [][] A2 = relu(Z2);
		/* Z3 = W3*A2 + bias_3*/
		double [][] Z3 = Matrix_add(Matrix_dot(w3,A2),b3);
		double [][] A3 = Z3;
		cost = compute_cost(A3,Y);
		
		/*back prop*/


		/* dZ3 = (A3-Y[0])*/
		double [][]dZ3 = Matrix_sub(A3,Y);
		/*dW3 = np.dot(dZ3,A2.T)*/
		double [][]dW3 = Matrix_dot(dZ3,Matrix_trans(A2));
		double [][]db3 = dZ3;
		/* dZ2 = np.dot(W3.T,dZ3)*relu_deriv(A2)*/
		double [][]dZ2 = Matrix_mul(Matrix_dot(Matrix_trans(w3),dZ3),relu_deriv(A2));
		/* dW2 = np.dot(dZ2,A1.T)*/
		double [][]dW2 = Matrix_dot(dZ2,Matrix_trans(A1));
		//v22 = update_v(w2,dW2,mu,v22);
		//w2 = update_weights(w2,v22);
		/* db2 = dZ2*/
		double [][]db2 = dZ2 ;
		//v44 = update_v(b2,db2,mu,v44);
		//b2 = update_weights(b2,v44);
		/*dZ1 = np.dot(W2.T,dZ2)*(A1)*(1-A1)*/
		double [][]dZ1 = Matrix_mul(Matrix_dot(Matrix_trans(w2),dZ2),relu_deriv(A1));
		/*dW1 = np.dot(dZ1,X.T)*/
		double [][]dW1 = Matrix_dot(dZ1,Matrix_trans(X));
		//v11 = update_v(w1,dW1,mu,v11);
		//w1 = update_weights(w1,v11);

		/*db1 = dZ1*/
		double [][]db1 = dZ1 ;
		v55 = update_v(w3,dW3,mu,v55);
		w3 = update_weights(w3,v55);
		v66 = update_v(b3,db3,mu,v66);
		b3 = update_weights(b3,v66);
		v22 = update_v(w2,dW2,mu,v22);
		w2 = update_weights(w2,v22);
		v44 = update_v(b2,db2,mu,v44);
		b2 = update_weights(b2,v44);
		v33 = update_v(b1,db1,mu,v33);
		b1 = update_weights(b1,v33);
		v11 = update_v(w1,dW1,mu,v11);
		w1 = update_weights(w1,v11);

		
		weights_1 = w1 ;
		weights_2 = w2;
		weights_3 = w3;
		bias_1 = b1;
		bias_2 = b2;
		bias_3 = b3;
		v1 = v11;
		v2 = v22;
		v3=  v33;
		v4 = v44;
		v5 = v55;
		v6 = v66;
		return cost;
	}
	// get the index of state 
	static public int index( int x, int y, int distance , int bearing ){
		int index = 0;
		index = x*Bearing*Distance*Y_coor+
		y*Distance*Bearing+distance*Bearing+bearing;
		return index;
	} // working 
	
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
	public static double [][] relu(double [][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				/*perform relu activation  for every element in the matrix  max(0,x)*/
				results[i][j] = Math.max(0,m[i][j]);
			}
		}
		return results;
		
	}
	public static double [][] relu_deriv(double [][]m){
		int row_num = m.length;
		int col_num = m[0].length;
		double [][] results = new double[row_num][col_num];
		for(int i=0; i<row_num; i++) {
			
			for(int j=0; j<col_num; j++) {
				/*perform relu activation derivative   for every element in the matrix  for output <0 , return 0 for output >0 return 1*/
				if(m[i][j]<=0) {
				results[i][j] = 0;
				}
				else {
					results[i][j] = 1;
				}
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
		return Matrix_add(Matrix_mul(mu_matrix,v),lr_product(learning_rate,dw));
		
	}
	public static double [][] update_weights(double[][]w,double[][]v){

		return Matrix_add(w,v);
	}
	
	public static void write (String file, double[][]weights) throws IOException{
	    try {
		BufferedWriter outputWriter = null;
	    outputWriter = new BufferedWriter(new FileWriter(file));
	    for (int i=0;i<4;i++) {
	    	for (int j=0;j<2;j++) {
		    outputWriter.write(String.valueOf(weights[i][j]));
		    outputWriter.newLine();
	    	}
	    }

	    outputWriter.flush();  
	    outputWriter.close(); 
	}
	    catch(Exception e) {
	    	
	    }
}
	
	public static double[][] loadWeights(File file,double[][] weights){
		
		BufferedReader r = null;
		double [][] results = new double [weights.length][weights[0].length];
		
		try{
			r = new BufferedReader(new FileReader(file));
			for (int i = 0; i < weights.length; i++)
				for (int j = 0; j < weights[0].length; j++)
					results[i][j] = Double.parseDouble(r.readLine());
		}
		catch (IOException e) {
			System.out.println("IOException trying to open reader: " + e);

		}
			catch (NumberFormatException e) {

		}
		catch (NullPointerException e) {
	}
		finally {
			try {
				if (r != null)
					r.close();
			}
			catch (IOException e) {
				System.out.println("IOException trying to close reader: " + e);
			}
		}
		return results;
	} // working oad function works

	
	
}
