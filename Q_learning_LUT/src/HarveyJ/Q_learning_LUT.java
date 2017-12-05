package HarveyJ;

import java.awt.geom.Point2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.RobocodeFileOutputStream;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;


public class Q_learning_LUT extends AdvancedRobot {
	public double PI = Math.PI;
	// state representations will be a 4 tuple{x_a,y_a,distance_to_enemy,Bearing}
	// below is the quantized states
	public static int X_coor = 8;
	public static int Y_coor = 6;
	public static int Distance = 10;
	public static int Bearing = 4;
	double getBearing ;
	public int current_state ;
	public int next_state ;
	double current_q = 0.0;
	double next_q = 0.0;
	double reward = 0.0;
	double total_reward_per_action = 0.0;
	double cum_reward;
	double [] reward_array = new double [4500];   // record rewards for multiple battles
	public static double [] win_rate = new double [1500];
	public static int win_count = 0;
	public static int index_win = 0;
	// hyper paramaters 
	double alpha = 0.15;  // learning rate
	double gamma = 0.9;  // discount factor
	double epsilon = 0.0;
	
	static int row_num = 8*6*10*4;
	static int col_num = 5;
	boolean initialize = false;
	boolean learning = true ;
	boolean offpolicy = true;
	boolean synchrnous = true;
	public static double [][] Q_table = new double [row_num][col_num];  // This Q_table is a String matrix, use this to save Q_table on disk 
	double [][] Q_table_double = new double[row_num][col_num]; // This Q_table is a double matrix , use this to perform numeric operations 
	public static int index1 = 0;
	private int previousState;
	private int previousAction;
	public static Enemy enemy ;
	
	public class Enemy{
		public int y;
		public int x;
		public int distance;
		public int bearing;
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
public void onRoundEnded(RoundEndedEvent e_1) {			
	System.out.println("win rate is  !!!" + win_count/((double)getRoundNum()+1.0));
	if(getRoundNum() % 10 ==0) {
		win_rate[index_win] = win_count/((double)getRoundNum()+1);
		System.out.println("win rate array is" + Arrays.toString(win_rate));	
		index_win +=1;
		saveData(getDataFile("win_rate.txt"));
	}
	index1=index1+1;


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
		loadData(getDataFile("LUT.txt"));
		
		turnGunRight(360); // initial scan 
		while(true) {
			if (learning == false) {
				int action = randInt(0,4);
				take_action(action);
				turnGunRight(360);
			}
			else {
				
			//Q_learning starts
			
			//step 1 get initial state ----works
			current_state = getState();
			System.out.println("current_state 		 "+ current_state);
			//step 2  find the action that would result in maximum Q_value    ---- works
			int action = choose_action(current_state);

			if (offpolicy == true) {
				next_q = max((Q_table[current_state]));
			}
			else {
				System.out.println("on policy");
				next_q = Q_table[current_state][action];
			}
			Q_table[previousState][previousAction] += alpha*(total_reward_per_action+gamma*next_q-Q_table[previousState][previousAction]);
			previousState = current_state;
			previousAction = action;
			total_reward_per_action = 0.0;
			take_action(action);
			// step 5 , after taking action , register the new state ---- works
			turnGunRight(360);
			execute();
		}

		}
	}

	public void onScannedRobot(ScannedRobotEvent e) {

		enemy.x = quantize_position(getX());
		enemy.y = quantize_position(getY());
		enemy.distance = quantize_distance(e.getDistance());
		enemy.bearing = quantize_bearing(e.getBearing());
		fire(3);

		System.out.println("x coor is " + enemy.x);
		
}
	public void onWin(WinEvent event) {
		System.out.println("winning !!!");
		win_count += 1;
		System.out.println("win count is " + win_count);
		saveTable(getDataFile("LUT.txt"));
		
		
		
	}
	public void onDeath(DeathEvent event){
		saveTable(getDataFile("LUT.txt"));
	}
	
	// get the current state
	public int getState(){

		return index(enemy.x,enemy.y,enemy.distance,enemy.bearing);
	}
	
	// get the index of state 
	static public int index( int x, int y, int distance , int bearing ){
		int index = 0;
		index = x*Bearing*Distance*Y_coor+
		y*Distance*Bearing+distance*Bearing+bearing;
		return index;
	} // working 

	
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
	public  int choose_action(int state){
		if(Math.random()>epsilon) {
			return  argmax(state);
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
}
