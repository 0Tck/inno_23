import { IsNotEmpty } from 'class-validator';

export class CreateSampleDto {
  id?: number;

  @IsNotEmpty()
  columnOne: string;

  @IsNotEmpty()
  columnTwo: string;
}
