package  
{
	
	import flash.display.MovieClip;
	import fl.text.TLFTextField;
	import flash.display.DisplayObject;
	import fl.controls.UIScrollBar;
	import fl.events.ComponentEvent;
	import flash.text.TextField;
	
	
	public class InfoBase extends MovieClip 
	{
		private const preferredWidth:uint = 1024;
		private const preferredHeight:uint = 768;
		private const sliderMargin:uint = 25;
		
		private var titleText:TextField;
		private var descText:TextField;
		private var scrollBar:UIScrollBar;
		
		public function InfoBase() 
		{
			super();
			
			this.titleText = new TextField();
			this.descText = new TextField();
			this.scrollBar = new UIScrollBar();
			
			Initialize();
		}
		
		private function Initialize() : void
		{
			titleText.x = 40;
			titleText.y = 20;
			titleText.text = "Some title";
			
			descText.x = 0;
			descText.y = 150;
			descText.height = this.preferredHeight - 225;
			descText.width = this.preferredWidth;

			descText.text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas condimentum varius dui vitae tincidunt." + 
								"Etiam tempus eros in ex imperdiet, eget auctor massa venenatis." + 
								"Maecenas lacinia ante quis tortor auctor pharetra. " +
								"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas condimentum varius dui vitae tincidunt." + 
								"Etiam tempus eros in ex imperdiet, eget auctor massa venenatis." + 
								"Maecenas lacinia ante quis tortor auctor pharetra. ";
								
			
			//this.scrollBar.addEventListener(ComponentEvent.MOVE, ScrollBarMoveHandler);
			this.scrollBar.scrollTarget = this.descText;
			this.scrollBar.direction = "horizontal";
			this.scrollBar.setSize(this.descText.width - 2 * sliderMargin, this.descText.height);
			this.scrollBar.move(descText.x + sliderMargin, descText.y  + 25);
			
			this.addChild(titleText);
			this.addChild(descText);
			this.addChild(scrollBar);
		}
		
		private function ScrollBarMoveHandler(e:ComponentEvent) : void
		{
			trace("dupa");
		}
	}
	
}
