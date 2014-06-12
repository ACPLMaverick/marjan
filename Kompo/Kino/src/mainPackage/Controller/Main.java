package mainPackage.Controller;

import mainPackage.Model.*;
import mainPackage.View.*;
public class Main {

	/**
	 * Metoda main programu
	 */
	public static void main(String[] args) {
		Controller theController = null;
		View theView = new View(theController);
		Model theModel = new Model(theController);
		theController = new Controller(theView, theModel);
		theView.setVisible(true);
	}
}
