/*
 * 
 */
package mainPackage.Controller;

import javax.swing.SwingUtilities;

import mainPackage.Model.*;
import mainPackage.View.*;
// TODO: Auto-generated Javadoc

/**
 * The Class Main.
 */
public class Main {

	/**
	 * Metoda main programu.
	 *
	 * @param args the arguments
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
	
	/**
	 * Log.
	 *
	 * @param log the log
	 */
	public static void log(String log)
	{
		System.out.println(log);
	}
}
