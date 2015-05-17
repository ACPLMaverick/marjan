package  
{
	import flash.sampler.StackFrame;
	import flash.display.Stage;
	import flash.events.MouseEvent;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import flash.ui.Mouse;
	
	public class MouseController 
	{
		private const TIMER_SPEED:int = 500;
		private const TIMER_REPEAT:int = 1;
		
		private static var instance:MouseController;
		
		private var stage:Stage;
		
		
		private var timer:Timer;
		
		public var currentMouseX:Number = 0;
		public var currentMouseY:Number = 0;
		public var relativeMouseX:Number = 0;
		public var relativeMouseY:Number = 0;
		public var isPressed:Boolean = false;
		public var isHold:Boolean = false;
		public var isClicked:Boolean = false;
		public var relativeScroll:int = 0;
		public var totalScroll:int = 0;
		
		public function Initialize(stage:Stage) : void
		{
			this.stage = stage;
			
			this.timer = new Timer(TIMER_SPEED, TIMER_REPEAT);
			
			this.stage.addEventListener(MouseEvent.MOUSE_DOWN, ButtonDownHandler);
			this.stage.addEventListener(MouseEvent.MOUSE_UP, ButtonUpHandler);
			this.stage.addEventListener(MouseEvent.MOUSE_WHEEL, ScrollHandler);
			this.stage.addEventListener(MouseEvent.RIGHT_MOUSE_DOWN, RightClickHandler);
		}
		
		public function Update() : void
		{
			this.relativeMouseX = stage.mouseX - currentMouseX;
			this.relativeMouseY = stage.mouseY - currentMouseY;
			this.currentMouseX = stage.mouseX;
			this.currentMouseY = stage.mouseY;
		}
		
		private function ButtonDownHandler(event:MouseEvent) : void
		{
			//trace("Down");
			this.isPressed = true;
			
			timer.reset();
			if(timer.hasEventListener(TimerEvent.TIMER_COMPLETE)) timer.removeEventListener(TimerEvent.TIMER_COMPLETE, TimerHandler);
			timer.addEventListener(TimerEvent.TIMER_COMPLETE, TimerHandler, false, 0, true);
			timer.start();
		}
		
		private function ButtonUpHandler(event:MouseEvent) : void
		{
			//trace("Up");
			
			if(!this.isHold)
			{
				System.getInstance().DispatchClick(null);
			}
			
			this.isPressed = false;
			this.isHold = false;
			timer.reset();
		}
		
		private function RightClickHandler(event:MouseEvent) : void
		{
			if(System.getInstance().IsCameraMoved)
			{
				System.getInstance().CameraSingleUnmove(300);
				return;
			}
			
			if(System.getInstance().CurrentMode > System.MODE_SELECTION)
			{
				System.getInstance().SetMode(System.getInstance().CurrentMode - 1);
			}
		}
		
		private function ScrollHandler(event:MouseEvent) : void
		{
			this.relativeScroll = event.delta;
			this.totalScroll += this.relativeScroll;
		}
		
		private function TimerHandler(event:TimerEvent) : void
		{
			//trace("Hold");
			this.isHold = true;
			System.getInstance().DispatchHold(null);
		}
		
		public static function getInstance() : MouseController
		{
			if(MouseController.instance == null)
			{
				MouseController.instance = new MouseController();
			}
			return MouseController.instance;
		}
	}
	
}
