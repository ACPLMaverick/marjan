package mainPackage.Controller;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.ListSelectionModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import mainPackage.Model.Model;
import mainPackage.Model.Repertoire;
import mainPackage.Model.Seance;
import mainPackage.Model.Ticket;
import mainPackage.View.View;

public class Controller {
	private View theView;
	private Model theModel;
	
	private SelectionListener sl = new SelectionListener();			//to na razie m�j jedyny pomys� jak pobra�
																	//zaznaczony tytu�
	
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
			theView.um.addBasketButtonListener(basketButtonListener);
			theView.um.disableButton(theView.um.getBasketButton());
			if(theModel.boughtTickets.get().isEmpty() == false)
			{
				theView.um.enableButton(theView.um.getBasketButton());
			}
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
			theView.um.buyTicket.getTicketCount().addChangeListener(spinnerChangeListener);
			theView.um.buyTicket.setSeanceTitle(sl.seance.getTitle(), sl.seance.getDateAsString());
			theView.um.buyTicket.setTicketPrice(sl.seance.getPrice());
			theView.um.buyTicket.addBuyButtonListener(buyTicketButtonListener);
		}
	};
	
	ChangeListener spinnerChangeListener = new ChangeListener(){
		@Override
		public void stateChanged(ChangeEvent arg0) {
			double cena = sl.seance.getPrice();
			cena *= Double.valueOf(theView.um.buyTicket.getSpinnerListModel().getValue().toString());
			theView.um.buyTicket.setTicketPrice(cena);
		}
	};
	
	ActionListener buyTicketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.createSmallWindow("Zakup trafi� do koszyka");
			for(int i = 0; i<Integer.valueOf(theView.um.buyTicket.getSpinnerListModel().getValue().toString()); i++){
				theModel.boughtTickets.add(new Ticket(sl.seance));
			}
			theView.um.enableButton(theView.um.getBasketButton());
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
	
	ActionListener basketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			SelectionController updater = new TicketsSelectionController(theModel.boughtTickets, null, null, null, 0, 0);
			Object[][] tickets = updater.getCollectionAsObjects();
			theView.um.createBasketMenu(tickets, null);
		}
	};
	
	class SelectionListener implements ListSelectionListener {
		public Seance seance;
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
			seance = theModel.repertoire.get(row);
		}
	};
	
	public void updateRepertoireTable()
	{
		SelectionController updater = new RepertoireSelectionController(theModel.repertoire, null, null, null, null, 0, 0);
		Object[][] newContent = updater.getCollectionAsObjects();
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