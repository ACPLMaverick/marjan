package mainPackage.Controller;

import javax.swing.SwingUtilities;

import mainPackage.Model.*;
import mainPackage.View.*;
public class Main {

	/**
	 * Metoda main programu
	 */
	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable()
		 { 
			 public void run()
			 { 
				 Controller theController = null;
				 View theView = new View(theController);
				 Model theModel = new Model(theController);
				 theController = new Controller(theView, theModel);
				 theView.setController(theController);
			 } 
		 }); 
	}
	
	public static void log(String log)
	{
		System.out.println(log);
	}
}
