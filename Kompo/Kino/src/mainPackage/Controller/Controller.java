package mainPackage.Controller;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.ListSelectionModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import mainPackage.Model.Model;
import mainPackage.Model.Repertoire;
import mainPackage.View.View;

public class Controller {
	private View theView;
	private Model theModel;
	
	private SelectionListener sl = new SelectionListener();			//to na razie mój jedyny pomys³ jak pobraæ
																	//zaznaczony tytu³
	
	public Controller(View theView, Model theModel){
		this.theView = theView;
		this.theModel = theModel;
		
		this.theView.addUserButtonListener(userButtonListener);
		this.theView.addAdminButtonListener(adminButtonListener);
	}
	
	ActionListener userButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createUserMenu();
			theView.um.addBuyButtonListener(buyButtonListener);
			theView.um.disableButton(theView.um.getBuyButton());
			theView.um.addBookButtonListener(bookButtonListener);
			theView.um.disableButton(theView.um.getBookButton());
			theView.um.addBackButtonListener(backButtonListener);
			theView.um.getUserListSelection().addListSelectionListener(sl);
			updateRepertoireTable();
		}
	};
	
	ActionListener adminButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createAdminMenu();
		}
	};
	
	ActionListener buyButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.um.createBuyTicketMenu();
			theView.um.buyTicket.setSeanceTitle(sl.txt, sl.date);
			theView.um.buyTicket.setTicketPrice(sl.price);
			theView.um.buyTicket.getTicketCount().addChangeListener(spinnerChangeListener);
		}
	};
	
	ChangeListener spinnerChangeListener = new ChangeListener(){
		
		@Override
		public void stateChanged(ChangeEvent arg0) {
			System.out.println(theView.um.buyTicket.getTicketCount().getValue());
		}
	};
	
	ActionListener bookButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.um.createBookTicketMenu();
		}
	};
	
	ActionListener backButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(true);
			theView.um.getUserMenu().setVisible(false);
		}
	};
	
	class SelectionListener implements ListSelectionListener {
		public String txt, date, price;		//TODO: zapisywanie w tablicy objektów zazaczonego rzêdu
		@Override
		public void valueChanged(ListSelectionEvent e) {
			ListSelectionModel lsm = (ListSelectionModel)e.getSource();
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			int row = theView.um.getUserTable().getSelectedRow();
			if(row < 0) return;
			int col = theView.um.getUserTable().getSelectedColumn();
			if(col < 0) return;
			
			theView.um.enableButton(theView.um.getBuyButton());
			theView.um.enableButton(theView.um.getBookButton());
			txt = theView.um.getMyTableModel().getValueAt(row, 0).toString();
			date = theView.um.getMyTableModel().getValueAt(row, 1).toString() + " "
					+ theView.um.getMyTableModel().getValueAt(row, 2).toString();
			price = theView.um.getMyTableModel().getValueAt(row, 4).toString();
		}
	};
	
	public void updateRepertoireTable()
	{
		SelectionController updater = new SelectionController(theModel.repertoire);
		Object[][] newContent = updater.getRepertoireAsObjects();
		theView.um.setTableContent(newContent);
	}
	
	public void serialiseRepertoire()
	{
		SerializationController ser = new SerializationController<Repertoire>(theModel.repertoire);
		ser.serialize("D:\\repertuar.xml");
	}
	
	public void deserialiseRepertoire()
	{
		SerializationController ser = new SerializationController<Repertoire>(theModel.repertoire);
		theModel.repertoire = (Repertoire)ser.deserialize("D:\\repertuar.xml");
	}
}
