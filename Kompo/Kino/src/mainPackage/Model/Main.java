package mainPackage.Model;

import java.awt.EventQueue;

import mainPackage.Controller.Controller;
import mainPackage.View.View;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Controller theController = null;
		View theView = new View(theController);
		Model theModel = new Model(theController);
		theController = new Controller(theView, theModel);
		theView.setVisible(true);
	}
}
