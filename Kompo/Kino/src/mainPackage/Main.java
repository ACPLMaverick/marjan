package mainPackage;

import java.awt.EventQueue;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		View theView = new View();
		Data theModel = new Data();
		Controller theController = new Controller(theView, theModel);
		theView.setVisible(true);
	}
}
